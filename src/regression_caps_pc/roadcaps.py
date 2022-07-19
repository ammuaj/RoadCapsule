"""RoadCaps Trainer."""


import torch
import os
import glob
import json
import pickle
import random
from torchmetrics import SpearmanCorrCoef, PearsonCorrCoef
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from utils import create_numeric_mapping,loss_plot_write
from layers import ListModule, PrimaryCapsuleLayer, HigherCapsuleLayer


class RoadCaps(torch.nn.Module):

    def __init__(self, args, number_of_features, number_of_targets):
        super(RoadCaps, self).__init__()
        """
        :param args: Arguments object.
        :param number_of_features: Number of vertex features.
        :param number_of_targets: Number of classes.
        """
        self.args = args
        self.number_of_features = number_of_features
        #self.number_of_features = 1
        self.number_of_targets = number_of_targets
        self.dropout = self.args.dropout
        self._setup_layers()

    def _setup_base_layers(self):
        """
        Creating GCN layers.
        """
        self.base_layers = [GCNConv(self.number_of_features, self.args.gcn_filters)]
        for _ in range(self.args.gcn_layers-1):
            self.base_layers.append(GCNConv(self.args.gcn_filters, self.args.gcn_filters))
        #self.base_layers.append(GCNConv(self.args.gcn_filters, 1))

        #self.base_layers.append(torch.nn.Linear(self.args.gcn_filters*self.number_of_targets, self.number_of_targets))

        #self.base_layers.append(torch.nn.Linear(self.args.gcn_filters,self.number_of_targets))
        #self.base_layers.append(torch.nn.Linear(10,1)) #self.number_of_targets,self.number_of_targets))
        self.base_layers = ListModule(*self.base_layers)

    def _setup_fully_connected_layer(self,no_of_layer = 0):
        if no_of_layer > 0:
            self.fully_connected_layer = [torch.nn.Linear(self.args.capsule_dimensions*self.number_of_targets,self.number_of_targets)]
            for _ in range(no_of_layer-1):
                self.fully_connected_layer.append(torch.nn.Linear(self.number_of_targets,self.number_of_targets))
            self.fully_connected_layer = ListModule(*self.fully_connected_layer)

    def _setup_primary_capsules(self):
        """
        Creating primary capsules.
        """
        self.first_capsule = PrimaryCapsuleLayer(in_units=self.args.gcn_filters,
                                                 in_channels= 1,#self.args.gcn_layers,
                                                 num_units= 1,#self.args.gcn_layers,
                                                 capsule_dimensions=self.args.capsule_dimensions)


    def _setup_graph_capsules(self):
        """
        Creating graph capsules.
        """
        self.graph_capsule = HigherCapsuleLayer(self.args.gcn_layers,
                                                self.args.capsule_dimensions,
                                                self.args.number_of_capsules,
                                                self.args.capsule_dimensions)

    def _setup_higher_level_capsule(self):
        """
        Creating class capsules.
        """
        self.higher_capsule = HigherCapsuleLayer(self.args.capsule_dimensions,
                                                 self.args.number_of_capsules,
                                                 self.number_of_targets,
                                                 self.args.capsule_dimensions)

    def _setup_layers(self):

        self._setup_base_layers()
        self._setup_primary_capsules()
        self._setup_higher_level_capsule()
        self.number_of_FCL = 0


    def forward(self, data):

        features = data["features"]
        edges = data["edges"]
        hidden_representations = []

        first_time = True
        for i, layer in enumerate(self.base_layers):
            if i != len(self.base_layers):
                if i < len(self.base_layers)-0:

                    features = torch.nn.functional.relu(layer(features, edges))
                    features = torch.nn.functional.dropout(features,self.dropout)

                    hidden_representations.append(features)
                elif i >= len(self.base_layers)-0:

                    if first_time:
                        features = features.view(-1,self.args.gcn_filters*self.number_of_targets)
                        features = torch.nn.functional.relu(layer(features))

                    #if i==len(self.base_layers)-1:
                    if first_time==False:
                        features = layer(features)
                    first_time = False

                    hidden_representations.append(features)

        res  = hidden_representations[-1]
        #
        # #
        #hidden_representations = torch.cat(tuple(hidden_representations[-1]))
        hidden_representations = hidden_representations[-1].view(1, 1, self.args.gcn_filters, -1)
        first_capsule_output = self.first_capsule(hidden_representations)


        first_capsule_output = first_capsule_output.view(-1, 1*self.args.capsule_dimensions)

        graph_capsule_output =  first_capsule_output #self.graph_capsule(rescaled_first_capsule_output)
        reshaped_graph_capsule_output = graph_capsule_output.view(-1, self.args.capsule_dimensions,
                                                                   self.args.number_of_capsules)

        higher_capsule_output = self.higher_capsule(reshaped_graph_capsule_output)
        higher_capsule_output = higher_capsule_output.view(-1, self.number_of_targets*self.args.capsule_dimensions)
        higher_capsule_output = torch.mean(higher_capsule_output, dim=0).view(1,
                                                                            self.number_of_targets,
                                                                              self.args.capsule_dimensions)

        fcl_output = higher_capsule_output

        return fcl_output #, reconstruction_loss
        #return res


class RoadCapsTrainer(object):
    """
    RoadCaps training and scoring.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.setup_model()

    def enumerate_unique_labels_and_targets(self):
        """
        Enumerating the features and targets in order to setup weights later.
        """
        print("\nEnumerating feature and target values.\n")
        ending = "*.json"

        self.train_graph_paths = glob.glob(self.args.train_graph_folder+ending)
        self.test_graph_paths = glob.glob(self.args.test_graph_folder+ending)
        graph_paths = self.train_graph_paths + self.test_graph_paths
        random.shuffle(graph_paths)

        n = int(len(graph_paths)*0.7)
        self.train_graph_paths = graph_paths[:n]
        self.test_graph_paths = graph_paths[n:]
        print("total train_ Samples: ",len(self.train_graph_paths))
        print("total test samples: ",len(self.test_graph_paths))

        targets = set()
        features = set()

        for path in tqdm(graph_paths):
            data = json.load(open(path))
            if self.args.target_nodes==1 and  self.args.neighbor_nodes != 0:
                data = data['neighbors_'+str(self.args.neighbor_nodes)]
            target_label = "target_"+str(self.args.target_nodes)
            targets = targets.union(set(data[target_label]))
            features = features.union(set(data["travel_factor"]))
            #print("features-",features)

        self.target_map = create_numeric_mapping(targets)
        self.feature_map = create_numeric_mapping(features)

        self.number_of_features = self.args.gcn_features
        self.number_of_nodes = len(self.feature_map)
        self.number_of_targets = len(self.target_map)
        print("num of features: ",self.number_of_features)
        print("num of targets: ",self.number_of_targets)
        print("num of Nodes: ",self.number_of_nodes)

    def setup_model(self):
        """
        Enumerating labels and initializing a RoadCaps.
        """
        self.enumerate_unique_labels_and_targets()
        self.model = RoadCaps(self.args, self.number_of_features, self.number_of_targets)

    def create_batches(self):
        """
        Batching the graphs for training.
        """
        self.batches = []
        for i in range(0, len(self.train_graph_paths), self.args.batch_size):
            self.batches.append(self.train_graph_paths[i:i+self.args.batch_size])

    def create_data_dictionary(self, target, edges, features):
        """
        Creating a data dictionary.
        :param target: Target vector.
        :param edges: Edge list tensor.
        :param features: Feature tensor.
        """
        to_pass_forward = dict()
        to_pass_forward["target"] = target
        to_pass_forward["edges"] = edges
        to_pass_forward["features"] = features
        return to_pass_forward

    def create_target(self, data):
        """
        Target createn based on data dicionary.
        :param data: Data dictionary.
        :return : Target vector.
        """
        #return  torch.FloatTensor([0.0 if i != data["target"] else 1.0 for i in range(self.number_of_targets)])
        targets = np.zeros(self.number_of_targets)
        target_label = "target_"+str(self.number_of_targets)
        #print(target_label)
        for k in data[target_label].keys():
            targets[int(k)] = data[target_label][k] # scaled to 0 to 1000
        return  torch.FloatTensor(targets)

    def create_edges(self, data):
        """
        Create an edge matrix.
        :param data: Data dictionary.
        :return : Edge matrix.
        """
        edges = [[edge[0], edge[1]] for edge in data["edges"]]
        #edges = edges + [[edge[1], edge[0]] for edge in data["edges"]]
        #print("edges: ",edges)
        return torch.t(torch.LongTensor(edges))

    def create_features(self, data):
        """
        Create feature matrix.
        :param data: Data dictionary.
        :return features: Matrix of features.
        """
        feat_combination = True
        #if feat_combination:
         #   self.number_of_features = 1
        self.feature_used = '_feat_tf_'  #which feature is using i.e ft = means feature travel factor
        #features = np.ones((self.number_of_nodes, 1))
        features = np.zeros((self.number_of_nodes, self.number_of_features))



        # if self.number_of_features>=4 or feat_combination==True:
        for k in data["travel_factor"].keys():
            features[int(k),0]= data["travel_factor"][k]
        # #
        for k in data["indegree"].keys():
            features[int(k),1]= data["indegree"][k]
        # # # #
        # for k in data["outdegree"].keys():
        #     features[int(k),2]= data["outdegree"][k]
        # # #     for k in data["boundary_features"].keys():
        # # #         features[int(k),3]= data["boundary_features"][k]
        # # # #if self.number_of_features==5:
        # for k in data["geohash_density"].keys():
        #   features[int(k),0]= data["geohash_density"][k]
            # for k in data["x_cor"].keys():
            #     features[int(k),4]= data["x_cor"][k]
            # for k in data["y_cor"].keys():a
            #     features[int(k),5]= data["y_cor"][k]

        # if self.number_of_features==5:
        #     for k in data["boundary_features"].keys():
        #         features[int(k),0]= data["boundary_features"][k]
        #
        # if self.number_of_features==6:
        #     for k in data["boundary_features"].keys():
        #         features[int(k),0]= data["boundary_features"][k]
        #
        # if self.number_of_features==2:
        #     for k in data["features-2"].keys():
        #         features[int(k),0]= data["features-2"][k]
        #
        #     for k in data["features-3"].keys():
        #         features[int(k),1]= data["features-3"][k]
        #
        # #if self.number_of_features==4:
        #  #   for k in data["boundary_features"].keys():
        #   #      features[int(k),2]= data["boundary_features"][k]
        # if self.number_of_features==3:
        #     for k in data["labels"].keys():
        #         features[int(k),]= data["labels"][k]


        features = torch.FloatTensor(features)
        return features

    def create_input_data(self, path):

        data = json.load(open(path))
        if self.args.target_nodes==1 and  self.args.neighbor_nodes != 0:
                data = data['neighbors_'+str(self.args.neighbor_nodes)]
        target = self.create_target(data)
        edges = self.create_edges(data)
        features = self.create_features(data)
        #print("torch features size",features.size())
        to_pass_forward = self.create_data_dictionary(target, edges, features)
        return to_pass_forward

    def fit(self):

        print("\nTraining started.\n")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        loss_list = []
        list_cor = []

        list_pos_avg_cor = []
        list_neg_avg_cor = []
        self.list_test_loss =[]
        self.list_test_cor = []

        self.list_test_pos_cor = []
        self.list_test_neg_cor = []
        self.best_loss = float ('inf')
        self.best_model = self.model

        #loading urban and rural osmids
        self.urban_rural_analysis = True
        if self.urban_rural_analysis:
            path = '/Users/muaz/Thesis/GNN/CapsGNN/CA500nx/'
            file = open(path+'CA_rural_osmids.pickle', 'rb')
            self.rural_osmids = pickle.load(file)
            file.close()

            file = open(path+'CA_urban_osmids.pickle', 'rb')
            self.urban_osmids = pickle.load( file)
            file.close()

        good_cor = 0
        for e in tqdm(range(self.args.epochs), desc="Epochs: ", leave=True):

            max_cor = -100
            min_cor = 100
            list_pos_cor = []
            list_neg_cor = []
            random.shuffle(self.train_graph_paths)
            good_cor = 0
            self.create_batches()
            losses = 0
            average_loss=0

            list_train_loss = []
            all_target = []
            all_prediction = []
            self.steps = trange(len(self.batches), desc="Loss")
            for step in self.steps:
                accumulated_losses = 0
                spearman_cor_batch_avg = 0

                optimizer.zero_grad()
                batch = self.batches[step]
                for path in batch:
                    data = self.create_input_data(path)
                    #prediction, reconstruction_loss = self.model(data)
                    #print("features: ", data["target"])

                    # if self.urban_rural_analysis:
                    #     test_data = json.load(open(path))
                    # if self.args.target_nodes==1 and  self.args.neighbor_nodes != 0:
                    #         test_data = test_data['neighbors_'+str(self.args.neighbor_nodes)]
                    # target_osmid = int(test_data['target_osmid'])
                    #
                    # #elif target_osmid in self.rural_osmids:
                    #
                    # if str(target_osmid) not in test_data['osmid_to_nid'].keys():
                    #      os.remove(path)
                    #      print(path)
                    #      continue

                    prediction = self.model(data)

                    prediction = prediction.squeeze()
                    #print(prediction.size())
                    #print(recon_loss)

                    if self.number_of_targets==1:
                        prediction = torch.sum(prediction,dim=0)
                    elif self.model.number_of_FCL==0:
                        prediction = torch.sum(prediction,dim=1)

                    #print(prediction.size())
                    #print(self.number_of_targets)
                    prediction = prediction.reshape(self.number_of_targets)


                    target = data["target"]
                    all_target.append(target)
                    all_prediction.append(prediction)

                    #print(prediction.size(),target.size())
                    target = target.reshape(self.number_of_targets)

                    if self.args.loss_function=='MSE':
                        loss = torch.nn.MSELoss().forward(prediction,target)
                    elif self.args.loss_function == 'MAE':
                        loss = torch.nn.L1Loss().forward(target,prediction)
                    elif self.args.loss_function == 'HUBER':
                        loss = torch.nn.HuberLoss().forward(target,prediction)

                    else:
                        loss = torch.nn.L1Loss().forward(target,prediction)

                    accumulated_losses = accumulated_losses + loss
                accumulated_losses = accumulated_losses/len(batch)
                #spearman_cor_batch_avg = spearman_cor_batch_avg/len(batch)
                accumulated_losses_tot = accumulated_losses
                accumulated_losses_tot.backward()
                optimizer.step()
                losses = losses + accumulated_losses.item()
                # spear_cor_all += spearman_cor_batch_avg
                # average_spear_cor = spear_cor_all/(step+1)
                # average_cor = average_spear_cor
                average_loss = losses/(step + 1)

                list_train_loss.append(average_loss)
                #list_train_loss.append(loss.item())

                #perc_good_cor = (good_cor/((step+1)*self.args.batch_size))*100

                #self.steps.set_description("RoadCaps (Loss=%.10f) (max_cor = %f) (min_cor = %f) (good_cor = %d ) (pos_avg_cor = %f) (neg_avg_cor = %f) ( tot_avg_Cor=%f)" % (round(average_loss, 10),max_cor,min_cor, good_cor,np.mean(list_pos_cor),np.mean(list_neg_cor),average_cor))
                self.steps.set_description(" Train Loss = %f" %round(average_loss, 10))
            #spearman =  SpearmanCorrCoef()
            #pearson = PearsonCorrCoef() #
            #cor = pearson(torch.FloatTensor(all_target),torch.FloatTensor(all_prediction))
            #print("Train Cor = %f" %cor)
            #spearman_cor_batch_avg=spearman_cor
            # average_cor = spearman_cor
            #
            # if spearman_cor >=0:
            #     list_pos_cor.append(spearman_cor.cpu().detach().numpy())
            # else:
            #     list_neg_cor.append(spearman_cor.cpu().detach().numpy())
            #
            # if spearman_cor > max_cor:
            #     max_cor = spearman_cor
            # if spearman_cor>=0.5 or spearman_cor <=-0.5:
            #     good_cor+=1
            # if spearman_cor >0 and spearman_cor < min_cor:
            #     min_cor = spearman_cor

            loss_list.append(average_loss)
            #list_cor.append(cor.cpu().detach().numpy())
            #list_pos_avg_cor.append(np.mean(list_pos_cor))
            #list_neg_avg_cor.append(np.mean(list_neg_cor))

             # saving best model
            if self.best_loss>average_loss:
                self.best_loss = average_loss
                self.best_model = self.model

            #print("(max_cor = %f) (min_cor = %f) (good_cor = %d ) (pos_avg_cor = %f) (neg_avg_cor = %f) ( tot_avg_Cor=%f)" % (max_cor,min_cor, good_cor,np.mean(list_pos_cor),np.mean(list_neg_cor),average_cor))

            #outPath = './graphSize100/output/'
            writePath = self.args.prediction_path
            #writing loss plot
            dataset_name= 'CA-500'#'California_500'
            loss_plot_write(writePath,loss_list,dataset_name+"target_"+str(self.number_of_targets)+"_neighbors_"+str(self.args.neighbor_nodes)+"_train_"+self.args.loss_function+"_loss"+"_GCN_layer_"+str(self.args.gcn_layers)+self.feature_used)

            self.train_list_loss = loss_list
            #plotting loss for all the batches of a single epoch
            if e<3:
                loss_plot_write(writePath,list_train_loss,dataset_name+"target_"+str(self.number_of_targets)+"_neighbors_"+str(self.args.neighbor_nodes)+"_train_"+self.args.loss_function+"_loss_epoch_"+str(e)+"_GCN_layer_"+str(self.args.gcn_layers)+self.feature_used,x_label = 'step',y_label= 'loss')
            if e==0:
                self.train_loss_first_epoch = list_train_loss
      #saving the model
            save_model = True
            if save_model:
                torch.save(self.best_model,writePath+'trained_model.pt')

           # Scoring on test data ....
            self.score(e)
            #self.save_predictions()
            self.model.train()


            # writing correlation plot
            #loss_plot_write(writePath,list_cor,dataset_name+"_Average_Correlation",y_label='correlation')
            #loss_plot_write(writePath,list_pos_avg_cor,dataset_name+"_Positive_Correlation",y_label='correlation')
            #loss_plot_write(writePath,list_neg_avg_cor,dataset_name+"_Negative_Correlation",y_label='correlation')

    def test_mse(self):
            """
            Scoring on the test set.
            """
            print("\n\nFinal Scoring on Test data...\n")
            self.model.eval()
            self.predictions = []
            self.list_mse = []
            self.best_test_sample_path = []
            self.diff =[]
            self.correlatoin = []


            self.lowest_loss = 999;
            for path in tqdm(self.test_graph_paths):
                data = self.create_input_data(path)
                prediction = self.model(data)
                prediction = prediction.reshape(self.number_of_targets)
                target = data["target"]
                target = target.reshape(self.number_of_targets)
                mse_loss = torch.nn.MSELoss().forward(prediction,target)
                abs_loss = torch.nn.L1Loss().forward(prediction,target)
                pearson_cor = PearsonCorrCoef()
                cor = pearson_cor(prediction,target)
                self.correlatoin.append(cor.cpu().detach().numpy())
                loss = mse_loss
                #if loss<self.lowest_loss:
                self.lowest_loss=loss #.cpu().detach().numpy()
                self.best_test_sample_path.append(path)
                #self.predictions.append(prediction.cpu().detach().numpy())
                self.diff.append(torch.abs(target-prediction).cpu().detach().numpy())

                self.list_mse.append(loss.cpu().detach().numpy())
           # print(f"MSE Score is : {np.mean(np.array(self.list_mse))} and std: {np.std(np.array(self.list_mse))} and Cor: {np.mean(np.array(self.correlatoin))}")
            print(f"MSE Score is : %0.10f and Cor : %0.10f" %( np.mean(np.array(self.list_mse)),  np.mean(np.array(self.correlatoin))))



    def score(self,epoch=1):
        """
        Scoring on the test set.
        """
        #print("\n\nScoring on test set \n")
        self.model.eval()
        self.test_predictions = []
        self.test_targets = []
        self.test_mse = []
        self.test_rural_loss = []
        self.test_urban_loss = []
        #self.test_cor = []
        self.test_pos_cor = []
        self.test_neg_cor = []
        self.test_diff = []
        self.test_sample_paths = []
        self.target_osmid = []
        max_cor = 0
        good_cor = 0

        with torch.no_grad():
            for path in tqdm(self.test_graph_paths):
                data = self.create_input_data(path)
                #prediction = torch.sum(self.model(data),0)

                prediction = self.model(data)
                prediction = prediction.squeeze()
                #print(prediction.size())

                if self.number_of_targets==1:
                    prediction = torch.sum(prediction,dim=0)
                elif self.model.number_of_FCL==0:
                    prediction = torch.sum(prediction,dim=1)
                #print(prediction.size())
                # prediction_mag = torch.sqrt((prediction**2).sum(dim=2))
                # _, prediction_max_index = prediction_mag.max(dim=1)
                # prediction = prediction_max_index.data.view(-1).item()
                #self.predictions.append(prediction)

                target = data["target"]
                target = target.reshape(self.number_of_targets)
                prediction = prediction.reshape(self.number_of_targets)
                # mse_loss = torch.nn.MSELoss().forward(prediction,target)
                # abs_loss = torch.nn.L1Loss().forward(target,prediction)
                # loss = abs_loss
                #pearson_cor = SpearmanCorrCoef() #PearsonCorrCoef()
                #cor = pearson_cor(prediction,target)


                if self.args.loss_function=='MSE':
                    loss = torch.nn.MSELoss().forward(prediction,target)
                elif self.args.loss_function == 'MAE':
                    loss = torch.nn.L1Loss().forward(target,prediction)
                elif self.args.loss_function == 'HUBER':
                    loss = torch.nn.HuberLoss().forward(target,prediction)
                else:
                    loss = torch.nn.L1Loss().forward(target,prediction)

                self.test_targets.append(target.cpu().detach().numpy())
                self.test_predictions.append((prediction.cpu().detach().numpy()))

                # if cor>0:
                #     self.test_pos_cor.append(cor.cpu().detach().numpy())
                # else:
                #     self.test_neg_cor.append(cor.cpu().detach().numpy())

                self.test_mse.append(loss.cpu().detach().numpy())
                self.test_diff.append(torch.abs(target-prediction).cpu().detach().numpy())
                self.test_sample_paths.append(path)

                if self.urban_rural_analysis:
                    test_data = json.load(open(path))
                    if self.args.target_nodes==1 and  self.args.neighbor_nodes != 0:
                            test_data = test_data['neighbors_'+str(self.args.neighbor_nodes)]
                    target_osmid = int(test_data['target_osmid'])
                    # if target_osmid in self.urban_osmids:
                    #     #print("found urban id "+str(target_osmid))
                    #     self.test_urban_loss.append(loss.cpu().detach().numpy())
                    #elif target_osmid in self.rural_osmids:

                    if self.number_of_targets==1:
                        self.target_osmid.append(target_osmid)
                    if str(target_osmid) in test_data['osmid_to_nid'].keys():
                        #os.remove(path)
                        #print(path)

                        nid = test_data['osmid_to_nid'][str(target_osmid)]
                        #print(nid)
                        geohasdensity = test_data['geohash_density'][str(nid)]

                        if geohasdensity < 0.2 and loss <500: #and len(self.test_rural_loss) <= 100:
                            self.test_rural_loss.append(loss.cpu().detach().numpy())
                        elif geohasdensity>0.45: # and len(self.test_urban_loss) <= 100:
                             self.test_urban_loss.append(loss.cpu().detach().numpy())


                #self.test_predictions.append(prediction.cpu().detach().numpy())

                # if cor > max_cor:
                #         max_cor = cor
                # if cor>=0.5 or cor <=-0.5:
                #         good_cor+=1

            #pearson_cor =PearsonCorrCoef()  #SpearmanCorrCoef() #
            #cor = pearson_cor(torch.FloatTensor(self.test_targets),torch.FloatTensor(self.test_predictions))
            #self.list_test_cor.append(cor.cpu().detach().numpy())
            print("\ntest_loss: " + str(round(np.mean(self.test_mse), 6))) #+ " test_correlation: "+str(cor.cpu().detach().numpy()))
            #print(f"Test_MSE Score is : %0.10f, max_cor: %f, good_cor: %d avg_pos_cor : %f avg_neg_cor : %f Avg_Cor : %0.10f" %( np.mean(self.test_mse), max_cor,good_cor, np.mean(self.test_pos_cor),np.mean(self.test_neg_cor), np.mean(self.test_cor)))
            #print(f"Test_Score : test_cor = %d, num_sam_neg_cor=%d" %(len(self.test_pos_cor),len(self.test_neg_cor)))
            self.list_test_loss.append(round(np.mean(self.test_mse), 6))
            #self.list_test_cor.append(round(np.mean(self.test_cor),6))

            #self.list_test_pos_cor.append(round(np.mean(self.test_pos_cor),6))
            #self.list_test_neg_cor.append(round(np.mean(self.test_neg_cor),6))


            plot_score = True
            writePath = self.args.prediction_path
            if plot_score:
                dataset_name= 'CA_500'#'California_500'
                loss_plot_write(writePath,self.test_mse,dataset_name+"target_"+str(self.number_of_targets)+"_neighbors_"+str(self.args.neighbor_nodes)+"_Test_"+self.args.loss_function+"_Loss_all_Samples_"+"_GCN_layer_"+str(self.args.gcn_layers)+self.feature_used,x_label='sample')

                if self.urban_rural_analysis:
                    self.test_urban_loss = sorted(self.test_urban_loss[-50:])
                    self.test_rural_loss = sorted(self.test_rural_loss)[-50:]
                    loss_plot_write(writePath,self.test_urban_loss,dataset_name+"target_"+str(self.number_of_targets)+"_neighbors_"+str(self.args.neighbor_nodes)+"_Test_"+self.args.loss_function+"_urban_Loss_all_Samples_"+"_GCN_layer_"+str(self.args.gcn_layers)+self.feature_used,x_label='sample')

                    loss_plot_write(writePath,self.test_rural_loss,dataset_name+"target_"+str(self.number_of_targets)+"_neighbors_"+str(self.args.neighbor_nodes)+"_Test_"+self.args.loss_function+"_rural_Loss_all_Samples_"+"_GCN_layer_"+str(self.args.gcn_layers)+self.feature_used,x_label='sample')

                # writing correlation plot
                #loss_plot_write(writePath,self.list_test_cor,dataset_name+"_Test_Correlation_all_Samples_",x_label='sample',y_label='correlation')

                loss_plot_write(writePath,self.list_test_loss,dataset_name+"target_"+str(self.number_of_targets)+"_neighbors_"+str(self.args.neighbor_nodes)+"_Test_"+self.args.loss_function+"_Loss"+"_GCN_layer_"+str(self.args.gcn_layers)+self.feature_used)
                # writing correlation plot
               # loss_plot_write(writePath,self.list_test_pos_cor,dataset_name+"_Test_Pos_Correlation",y_label='correlation')
                #loss_plot_write(writePath,self.list_test_neg_cor,dataset_name+"_Test_Neg_Correlation",y_label='correlation')
                #plotting prediction differences of each node
                #for i in range(5):
                 #   loss_plot_write(writePath,self.test_diff[i],dataset_name+"_Test_differences_samp_"+str(i),x_label='node',y_label='bc_diff')


            file = open(self.args.prediction_path+'_model:GCNCAPS_test_prediction_for_targets'+str(self.number_of_targets)+'_neighbors_'+str(self.args.neighbor_nodes)+"_"+ self.args.loss_function + 'loss'+"_GCN_layer_"+str(self.args.gcn_layers)+self.feature_used+'.pickle', 'wb')

            pickle.dump([self.target_osmid,self.test_targets,self.test_predictions,self.test_mse,self.list_test_loss,self.train_loss_first_epoch,self.train_list_loss], file)
            #pickle.dump([self.test_targets,self.test_predictions,self.test_mse,self.list_test_loss,self.train_loss_first_epoch,self.train_list_loss,self.test_rural_loss,self.test_urban_loss], file)
    def save_predictions(self):
        """
        Saving the test set predictions.
        """

        #identifiers = path.split("/")[-1].strip(".json") # for path in self.test_graph_paths]
        output = {} #pd.DataFrame()
        for s in range(len(self.test_sample_paths)):
            path = self.test_sample_paths[s]
            data = json.load(open(path))
            out = {}
            out['sample_path']=path
            out['correlation'] = self.test_cor[s]
            out['mse_loss'] = self.test_mse[s]
            #out["id"] = identifiers
            prediction = {}
            pred_diff = {}
            target = {}

            original_ids = data['original_labels']
            targets = data['target']
            keys = list(original_ids.keys())
            for i in range(len(original_ids)):
                prediction[original_ids[keys[i]]] = self.predictions[s][i]
                pred_diff[original_ids[keys[i]]] = self.test_diff[s][i]
                target[original_ids[keys[i]]] = targets[str(i)]

            out["targets"] = target
            out["test_predictions"] = prediction
            out["prediction_difference"] = pred_diff

            #out.to_csv(self.args.prediction_path, index=None)
            #print(out)
            output[str(s)]=out

        #out to_csv(self.args.prediction_path, index=None)
        #nx.write_gpickle(list_ca_graph[:100],'CA_500_size_1ksamepls_new.pickle',pickle.DEFAULT_PROTOCOL)
        file = open(self.args.prediction_path+'test_predictions.pickle', 'wb')
        pickle.dump(output, file)
# dump yormation to that file

        #with open(self.args.prediction_path, 'w') as outfile:
         #   json.dump(out, outfile)



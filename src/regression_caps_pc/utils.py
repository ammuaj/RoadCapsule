"""Data reading and printing utils."""
from matplotlib.ticker import FormatStrFormatter
from texttable import Texttable
import random, os
import numpy as np
import torch

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from matplotlib.pyplot import figure

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def create_numeric_mapping(node_properties):
    """
    Create node feature map.
    :param node_properties: List of features sorted.
    :return : Feature numeric map.
    """
    return {value:i for i, value in enumerate(node_properties)}

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def loss_plot_write(write_path,list_loss,type='train_MSE',y_label = 'loss',x_label = 'epoch'):
    prefix = "" #+str(len(list_loss))#data_path.split("/")[-1]

            # for x,y in zip(xs,ys):

    plt.plot(list_loss, 'r-')


    plt.title("%s_%s_BC " % (prefix,type), fontsize=10)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=8, )
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.10f'))
    #plt.set_ylim([0.0, 1])
    #plt.set_yticks(np.arange(0, 1, 0.1), minor=False)
    # label = ["k=(5,10)", "k=(10,15)", "k=(15,5)", "k=(5,rank)", "k=(10, rank)", "k=(15,rank)"]
    # plt.annotate(label[x-1], # this is the text
    # (x,y),# these are the coordinates to position the label
    # textcoords="offset points", # how to position the text
    # xytext=(0,10), # distance from text to points (x,y)
    #  ha='center') # horizontal alignment can be left, right or center
    # plt.legend(loc="upper left")
    # plt.xticks(xs, label, fontsize=13)
        #plt.show()
    plt.savefig("%s%s_%s_plot.png"%(write_path, prefix, type), dpi=300, figsize=(15,10), bbox_inches='tight')
    plt.close()

# def loss_write():
#      if mse_test<best_mse_test:
#         file = open("FasBetModel.pickle","wb")
#         pickle.dump(model,file)
#         file.close()
#         best_mse_test = mse_test
#
#     prefix = data_path.split("/")[-1]
#     if e % 20 == 0:
#         generate_plots = True
#         if generate_plots:
#             # for x,y in zip(xs,ys):
#
#             plt.plot(test_mse_list, 'r-')
#
#             plt.title("%s_test_BC " % prefix, fontsize=18)
#             plt.xlabel('Epoch', fontsize=16)
#             plt.ylabel('MSE', fontsize=16)
#             # label = ["k=(5,10)", "k=(10,15)", "k=(15,5)", "k=(5,rank)", "k=(10, rank)", "k=(15,rank)"]
#             # plt.annotate(label[x-1], # this is the text
#             # (x,y),# these are the coordinates to position the label
#             # textcoords="offset points", # how to position the text
#             # xytext=(0,10), # distance from text to points (x,y)
#             #  ha='center') # horizontal alignment can be left, right or center
#             # plt.legend(loc="upper left")
#             # plt.xticks(xs, label, fontsize=13)
#         #plt.show()
#         plt.savefig("%s_test_BC_MSE.png" % prefix, dpi=300, bbox_inches='tight')
#     plt.close()

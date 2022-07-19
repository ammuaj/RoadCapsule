RoadCaps 
============================

### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          2.4
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             1.1.0
torch-scatter     1.4.0
torch-sparse      0.4.3
torch-cluster     1.4.5
torch-geometric   1.3.2
torchvision       0.3.0
```
### Datasets

### Outputs

### Options
<p align="justify">
Training a RoadCaps model is handled by the `src/main.py` script which provides the following command line arguments.</p>

#### Input and output options
```
  --training-graphs   STR    Training graphs folder.      Default is `dataset/train/`.
  --testing-graphs    STR    Testing graphs folder.       Default is `dataset/test/`.
  --prediction-path   STR    Output predictions file.     Default is `output/watts_predictions.csv`.
```
#### Model options
```
  --epochs                      INT     Number of epochs.                  
  --batch-size                  INT     Number fo graphs per batch.        
  --gcn-filters                 INT     Number of filters in GCNs.         
  --gcn-layers                  INT     Number of GCNs chained together.  
  --inner-attention-dimension   INT     Number of neurons in attention.     
  --capsule-dimensions          INT     Number of capsule neurons.         
  --number-of-capsules          INT     Number of capsules in layer.       
  --weight-decay                FLOAT   Weight decay of Adam.              
  --lambd                       FLOAT   Regularization parameter.          
  --learning-rate               FLOAT   Adam learning rate.                
```
### Examples
The following commands learn a model and save the predictions. Training a model on the default dataset:
```sh
$ python src/main.py
```
<p align="center">
  <img width="500" src="capsgnn.gif">
</p>

Training a CapsGNNN model for a 100 epochs.
```sh
$ python src/main.py --epochs 100
```

Changing the batch size.

```sh
$ python src/main.py --batch-size 128
```
----------------------

**License**

- [GNU License](https://github.com/benedekrozemberczki/CapsGNN/blob/master/LICENSE)

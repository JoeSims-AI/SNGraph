"""
    This script is for training the node classifier.
    This will create a new directory for the saved models, a directory for log files, a directory for loss files
    and confusion matrices (metrics).

    This code is only supposed to run for one fold. This is because the time limit on the HPC we were using lasted
    roughly the time it took to train one fold and nothing more.

    So the project directory should now look as shown below if all codes have been run so far.

    ├── Project
    │   ├── Graphs
    │   │   ├── SN0
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   │   ├── SN1
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   │   ├── SN2
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   ├── LogFiles
    │   ├── Models
    │   ├── Metrics

"""

# --------------------------------------------- Install packages -----------------------------------------------------

from glob import glob
import os
from os.path import isfile, isdir
import sys
import torch
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx, remove_self_loops, add_self_loops, degree
from torch.nn import Parameter, Sequential, Linear, ReLU, Tanh
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import math

from Utilities.default_args import get_params
from EdgeGraph.utils import order_files, get_id
from EdgeGraph.data import graph_object


# --------------------------------------------- Load project parameters ------------------------------------------------

params = get_params(sys.argv[1])
print('Got default params.')

# ------------------------------------------- Set up project directories -----------------------------------------------

name = params["NAME"]

if not isdir(params["MODEL_DIR"]):
    os.mkdir(params["MODEL_DIR"])
    print(f'Made directory {params["MODEL_DIR"]}')

if not isdir(params["METRIC_DIR"]):
    os.mkdir(params["METRIC_DIR"])
    print(f'Made directory {params["METRIC_DIR"]}')

if not isdir(params["LOG_DIR"]):
    os.mkdir(params["LOG_DIR"])
    print(f'Made directory {params["LOG_DIR"]}')

# ----------------------------------------------- Set up logger --------------------------------------------------------

log_no = len(os.listdir(params["LOG_DIR"]))
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(params["LOG_DIR"], f'{name}_{log_no}.log'), level=logging.DEBUG)
logger.info("Loaded packages and parameter file.")

# ----------------------------------- Set up additional parameters -----------------------------------------------------

in_features = list(params["IN_FEATURES"])
edge_features = list(params["EDGE_FEATURES"])
out_features = list(params["OUT_FEATURES"])
n_mps = int(params["N_MESSAGE_PASSING"])
h_features = 64 if "HIDDEN_FEATURES" not in params.keys() else int(params["HIDDEN_FEATURES"])

SN = int(params["SN"])

# ---------------------------------------------- Loading the Data ------------------------------------------------------

node_files, edge_files = order_files(params["NODE_DIR"],
                                     params["EDGE_DIR"])

train_test = pd.read_csv(params["CV_PATH"])
train_test['id'].astype(str)
if train_test['id'].dtype is not str:
    train_test['id'] = train_test['id'].astype(str)

logger.info("Loaded file paths and cross validation file.")

# ---------------------------------------- Set up metric tracking ------------------------------------------------------

loss_filename = os.path.join(params["METRIC_DIR"], f'{params["name"]}_loss.txt')
cm_filename = os.path.join(params["METRIC_DIR"], f'{params["name"]}_cm.txt')  # cm (confusion matrix)
if not isfile(loss_filename):
    loss_file = open(loss_filename, 'w+')
    loss_file.close()

if not isfile(cm_filename):
    cm_file = open(cm_filename, 'w+')
    cm_file.close()

# --------------------------------------------- Create Graphs ----------------------------------------------------------

train_graphs = []
val_graphs = []
for i, (node_file, edge_file) in enumerate(zip(node_files, edge_files)):
    tissue_id = get_id(node_file)
    if tissue_id in train_test['id'].tolist():
        split_set = train_test[train_test['id'] == tissue_id][f'fold_{params["FOLD"]}'].item()
        if split_set == 'train':
            train_graphs.append(graph_object(node_path=node_file,
                                             edge_path=edge_file,
                                             node_features=in_features,
                                             edge_features=edge_features,
                                             out_features=out_features,
                                             out_column='class',
                                             sn=params["SN"],
                                             threshold=params["THRESHOLD"],
                                             ))






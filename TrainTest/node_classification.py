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

import os
from os.path import isfile, isdir
import sys
import torch
import logging
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

from Utilities.default_args import get_params
from EdgeGraph.utils import order_files, get_id
from EdgeGraph.data import graph_object
from EdgeGraph.models import GraphConvMMP
from EdgeGraph.loss import weighted_mean_squared_error
from EdgeGraph.eval import balanced_acc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
logger.info(f"Device = {device}")

# ----------------------------------- Set up additional parameters -----------------------------------------------------

in_features = params["IN_FEATURES"]
edge_features = params["EDGE_FEATURES"]
out_features = params["OUT_FEATURES"]

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
    logger.info(f'Created {loss_filename}.')

if not isfile(cm_filename):
    cm_file = open(cm_filename, 'w+')
    cm_file.close()
    logger.info(f'Created {cm_filename}.')

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
                                             y=True,
                                             bin_out=False,
                                             ))
        elif split_set == 'val':
            val_graphs.append(graph_object(node_path=node_file,
                                           edge_path=edge_file,
                                           node_features=in_features,
                                           edge_features=edge_features,
                                           out_features=out_features,
                                           out_column='class',
                                           sn=params["SN"],
                                           threshold=params["THRESHOLD"],
                                           y=True,
                                           bin_out=False,
                                           ))
        else:
            continue

logger.info("Created graphs")
trainLoader = DataLoader(train_graphs, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"])
logger.info('\tTrain graphs =', len(train_graphs))

valLoader = DataLoader(val_graphs, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"])
logger.info('\tVal graphs =', len(val_graphs))

# ---------------------------------------- Calculate loss weights ------------------------------------------------------

loss_weights = [0] * len(out_features)
for graph in trainLoader:
    y = graph.y
    for i in range(len(loss_weights)):
        loss_weights[i] += torch.sum(y[:, i]).item()

norm_weights = [round(max(loss_weights) / weight, 4) for weight in loss_weights]
logger.info(f'Normalised loss weights: {norm_weights}')

# ----------------------------------------- Initialise the model -------------------------------------------------------

model = GraphConvMMP(in_features=len(in_features),
                     out_features=len(out_features),
                     edge_dims=len(edge_features),
                     n_message_passing=params["N_MESSAGE_PASSING"],
                     hidden_features=params["HIDDEN_FEATURES"],
                     mlp_features=64)

# ----------------------------- Load existing models and set up training parameters ------------------------------------

model_list = os.listdir(params["MODEL_DIR"])
if len(model_list) == 0:
    logger.info('Initialised new model.')
    pre_epochs = 0
else:
    model_epochs = [int(m.split('_')[-1].strip('.h5')) for m in model_list]
    model_name = model_list[model_epochs.index(max(model_epochs))]
    pre_epochs = max(model_epochs)
    print(model_name)
    try:
        model.load_state_dict(torch.load(model_name))
        logger.info('Matched model keys successfully.')
    except:
        raise Exception(f"Model does not exist: {model_name}")

model_name = os.path.join(params["MODEL_DIR"], f'{name}_fold{params["FOLD"]}')

momentum = 0.9 if "MOMENTUM" not in params.keys() else float(params["MOMENTUM"])
optimizer = torch.optim.SGD(list(model.parameters()), lr=params["LR"], momentum=momentum)

# ------------------------------------------------ Train Loop ----------------------------------------------------------

model = model.to(device)

if pre_epochs == 0:
    loss_file = open(loss_filename, 'a')
    loss_file.write(f'\nFold {params["FOLD"]}')
    loss_file.close()

losses = []
for epoch in range(1, params["EPOCHS"] + 1):
    batch_loss = []

    for n, graph in enumerate(trainLoader):

        # Attach batch graphs to device
        graph = graph.to(device)
        x = graph.x.type(torch.float)
        edge_index = graph.edge_index.type(torch.int)
        edge_attr = graph.edge_attr.type(torch.float)
        y_true = graph.y.type(torch.float)

        # Data Augmentation
        if params["NOISE_CELLS"] != 0:
            prob_ids = [in_features.index(in_f) for in_f in in_features if 'O' in in_f]
            r_cells = torch.randn(x.shape[0], len(prob_ids)) * float(params["NOISE_CELLS"])
            r_cells = r_cells.to(device)
            x[:, prob_ids] = torch.clamp(x[:, prob_ids] + r_cells, min=0)

        if params["NOISE_ATTR"] != 0 and params["NOISE_EDGE"] != 0:
            r_attr = torch.randn_like(edge_attr) * float(params["NOISE_ATTR"])
            r_attr = torch.exp(-r_attr + 1)
            r_attr = r_attr.to(device)
            edge_attr = torch.clamp(edge_attr + r_attr, min=0)

        if params["NOISE_ATTR"] != 0:
            r_attr = torch.exp(-r_attr)
            r_attr = r_attr.to(device)
            edge_attr = (edge_attr * r_attr)

        optimizer.zero_grad()
        output = model(x, edge_index, edge_attr)
        loss = weighted_mean_squared_error(y_true, output, norm_weights, device)
        batch_loss.append(loss.item())

        loss.backward(retain_graph=True)
        optimizer.step()

        del x
        del edge_index
        del edge_attr
        del y_true
        del loss
        del output
        if params["NOISE_CELLS"] != 0:
            del r_cells
            del prob_ids
        if params["NOISE_ATTR"] != 0:
            del r_attr

        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)

    logger.info(f'Loss Epoch {epoch} : {sum(batch_loss) / len(batch_loss)}')
    losses.append(sum(batch_loss) / len(batch_loss))

    if epoch % params["SAVE_EPOCHS"] == 0:

        # Save the model.
        torch.save(model.state_dict(), f'{model_name}_{pre_epochs + epoch}.h5')

        loss_file = open(loss_filename, 'a')
        loss_file.write('\n')
        loss_file.write('\n'.join(map(str, losses[-params["SAVE_EPOCHS"]:])))
        loss_file.close()

        # Get validation metrics for train data.
        confusion = np.zeros([len(out_features), len(out_features)])
        for n, graph in enumerate(trainLoader):
            graph = graph.to(device)
            x = graph.x.type(torch.float)
            edge_index = graph.edge_index.type(torch.int)
            edge_attr = graph.edge_attr.type(torch.float)
            y_true = graph.y.type(torch.float)
            with torch.no_grad():
                output = model(x, edge_index, edge_attr)
            output = output.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()

            max_out = np.argmax(output, axis=1)
            max_gt = np.argmax(y_true, axis=1)

            sum_gt = np.sum(y_true, axis=1)
            for i in range(0, output.shape[0]):
                if sum_gt[i] > 0:
                    confusion[max_out[i]][max_gt[i]] = confusion[max_out[i]][max_gt[i]] + 1

        correct = np.trace(confusion)
        tot = np.sum(np.sum(confusion, axis=1), axis=0)
        prop_correct = correct / tot
        bal_acc = balanced_acc(confusion)

        con_mat = open(cm_filename, 'a')
        con_mat.write(f'\nFold {params["FOLD"]} - {pre_epochs + epoch}')
        con_mat.write(f'\nTrain:\nAcc:\t{prop_correct:.6f}\nBal Acc:\t{bal_acc:.6f}\n')
        con_mat.close()

        # Get validation metrics for validation data.
        confusion = np.zeros([len(out_features), len(out_features)])
        for n, graph in enumerate(valLoader):
            graph = graph.to(device)
            x = graph.x.type(torch.float)
            edge_index = graph.edge_index.type(torch.int)
            edge_attr = graph.edge_attr.type(torch.float)
            y_true = graph.y.type(torch.float)
            with torch.no_grad():
                output = model(x, edge_index, edge_attr)
            output = output.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()

            max_out = np.argmax(output, axis=1)
            max_gt = np.argmax(y_true, axis=1)

            sum_gt = np.sum(y_true, axis=1)
            for i in range(0, output.shape[0]):
                if sum_gt[i] > 0:
                    confusion[max_out[i]][max_gt[i]] = confusion[max_out[i]][max_gt[i]] + 1

        correct = np.trace(confusion)
        tot = np.sum(np.sum(confusion, axis=1), axis=0)
        prop_correct = correct / tot
        bal_acc = balanced_acc(confusion)
        con_mat = open(cm_filename, 'a')
        con_mat.write(f'\nVal:\nAcc:\t{prop_correct:.6f}\nBal Acc:\t{bal_acc:.6f}\n')
        con_mat.close()

        del graph
        del x
        del edge_index
        del edge_attr
        del y_true
        del output
        del confusion
        del prop_correct
        del tot
        del correct

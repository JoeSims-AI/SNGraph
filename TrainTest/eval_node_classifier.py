"""
    Assuming all folds have been trained to the same number of epochs, this code goes through the train, validation
    and test set and records the accuracy and balanced accuracy across all folds and then calculates the mean and
    standard error over these folds. The output will be put in the "Metrics" directory.

    ├── Project
    │   ├── Graphs
    │   │   ├── SN0
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   │   ├── SN1
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   ├── LogFiles
    │   ├── Models
    │   ├── Metrics  <---- output will be in here

"""

# --------------------------------------------- Install packages -----------------------------------------------------

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

from Utilities.default_args import get_params
from EdgeGraph.utils import order_files, get_id
from EdgeGraph.data import graph_object
from EdgeGraph.models import GraphConvMMP
from EdgeGraph.eval import balanced_acc, format_cm, mean_acc

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# --------------------------------------------- Load project parameters ------------------------------------------------

params = get_params(sys.argv[1])
print('Got default params.')

name = params["NAME"]

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

print("Loaded file paths and cross validation file.")

# ------------------------------------ Create the evaluation file ------------------------------------------------------

acc_filename = os.path.join(params["METRIC_DIR"], f'{name}_{params["EPOCHS"]}_eval_metrics.txt')
acc_file = open(acc_filename, 'w+')
acc_file.close()

print(f'Created {acc_filename}.')


# --------------------------------------------- Create Graphs ----------------------------------------------------------


def get_folds(directory):
    models = [m.split('_')[-2] for m in os.listdir(directory)]
    model_folds = max([int(f.replace('fold', '')) for f in list(set(models))])
    return model_folds


n_folds = get_folds(params["MODEL_DIR"])

all_train_accs = []
all_train_baccs = []

all_val_accs = []
all_val_baccs = []

all_test_accs = []
all_test_baccs = []

# The evaluation loop over each fold begins here.
for fold_i in range(n_folds):

    divider = '-' * 50 + f'Fold {fold_i}' + '-' * 50

    acc_file = open(acc_filename, 'a')
    acc_file.write(divider)
    acc_file.close()

    # --------------------------------------------- Create Graphs ------------------------------------------------------

    train_graphs = []
    val_graphs = []
    test_graphs = []
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
                test_graphs.append(graph_object(node_path=node_file,
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

    print("Created graphs")
    trainLoader = DataLoader(train_graphs, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"])
    print('\tTrain graphs =', len(train_graphs))

    valLoader = DataLoader(val_graphs, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"])
    print('\tVal graphs =', len(val_graphs))

    testLoader = DataLoader(test_graphs, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"])
    print('\tTest graphs =', len(test_graphs))

    # ----------------------------------------- Initialise the model ---------------------------------------------------

    model = GraphConvMMP(in_features=len(in_features),
                         out_features=len(out_features),
                         edge_dims=len(edge_features),
                         n_message_passing=params["N_MESSAGE_PASSING"],
                         hidden_features=params["HIDDEN_FEATURES"],
                         mlp_features=64)

    # ------------------------------------------ Match model keys ------------------------------------------------------

    model_name = os.path.join(params["MODEL_DIR"], f'{name}_fold{fold_i}_{params["EPOCHS"]}.h5')

    try:
        model.load_state_dict(torch.load(model_name))
        print('Matched model keys successfully.')
    except:
        raise Exception(f"Model does not exist: {model_name}")

    model = model.to(device)

    # --------------------------------------- Evaluate Train Data ------------------------------------------------------

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

    if params["NORMALISE_CM"]:
        confusion = confusion / confusion.sum()

    correct = np.trace(confusion)
    tot = np.sum(np.sum(confusion, axis=1), axis=0)
    prop_correct = correct / tot
    bal_acc = balanced_acc(confusion)
    confusion = format_cm(confusion, dp=4)

    acc_file = open(acc_filename, 'a')
    acc_file.write('\nTrain\n')
    acc_file.write(confusion)
    acc_file.write(f'\nAccuracy {prop_correct}\n')
    acc_file.write(f'Balanced Accuracy {bal_acc}\n')
    acc_file.close()

    all_train_accs.append(prop_correct)
    all_train_baccs.append(bal_acc)

    # ---------------------------------------- Evaluate Val Data -------------------------------------------------------

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

    if params["NORMALISE_CM"]:
        confusion = confusion / confusion.sum()

    correct = np.trace(confusion)
    tot = np.sum(np.sum(confusion, axis=1), axis=0)
    prop_correct = correct / tot
    bal_acc = balanced_acc(confusion)
    confusion = format_cm(confusion, dp=4)

    acc_file = open(acc_filename, 'a')
    acc_file.write('\nVal\n')
    acc_file.write(confusion)
    acc_file.write(f'\nAccuracy {prop_correct}\n')
    acc_file.write(f'Balanced Accuracy {bal_acc}\n')
    acc_file.close()

    all_train_accs.append(prop_correct)
    all_train_baccs.append(bal_acc)

    # ---------------------------------------- Evaluate Test Data ------------------------------------------------------

    confusion = np.zeros([len(out_features), len(out_features)])
    for n, graph in enumerate(testLoader):
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

    if params["NORMALISE_CM"]:
        confusion = confusion / confusion.sum()

    correct = np.trace(confusion)
    tot = np.sum(np.sum(confusion, axis=1), axis=0)
    prop_correct = correct / tot
    bal_acc = balanced_acc(confusion)
    confusion = format_cm(confusion, dp=4)

    acc_file = open(acc_filename, 'a')
    acc_file.write('\nTest\n')
    acc_file.write(confusion)
    acc_file.write(f'\nAccuracy {prop_correct}\n')
    acc_file.write(f'Balanced Accuracy {bal_acc}\n\n')
    acc_file.close()

    all_train_accs.append(prop_correct)
    all_train_baccs.append(bal_acc)

# ---------------------------------------- Calculating the means -------------------------------------------------------

train_a, train_u = mean_acc(all_train_accs)
val_a, val_u = mean_acc(all_val_accs)
test_a, test_u = mean_acc(all_test_accs)

train_ba, train_bu = mean_acc(all_train_baccs)
val_ba, val_bu = mean_acc(all_val_baccs)
test_ba, test_bu = mean_acc(all_test_baccs)

train_a = f"Mean Train = {train_a:.5f} +- {train_u:.5f}"
val_a = f"Mean Val = {val_a:.5f} +- {val_u:.5f}"
test_a = f"Mean Test = {test_a:.5f} +- {test_u:.5f}"

train_ba = f"Mean Bal Train = {train_ba:.5f} +- {train_bu:.5f}"
val_ba = f"Mean Bal Val = {val_ba:.5f} +- {val_bu:.5f}"
test_ba = f"Mean Bal Test = {test_ba:.5f} +- {test_bu:.5f}"

print('')
print(train_a)
print(val_a)
print(test_a)
print('')
print(train_ba)
print(val_ba)
print(test_ba)

acc_file = open(acc_filename, 'a')
acc_file.write(train_a + '\n')
acc_file.write(val_a + '\n')
acc_file.write(test_a + '\n\n')
acc_file.write(train_ba + '\n')
acc_file.write(val_ba + '\n')
acc_file.write(test_ba + '\n\n')
acc_file.close()

print('Evaluation complete.')
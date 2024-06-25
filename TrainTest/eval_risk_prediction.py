"""
    Assuming all folds have been trained to the same number of epochs, this code goes through the train, validation
    and test set and records the c index across all folds and then calculates the mean and
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
from SNGraph.utils import order_files, get_id
from SNGraph.data import graph_object
from SNGraph.models import GraphAttSurv
from SNGraph.eval import calculate_cindex, mean_acc


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# --------------------------------------------- Load project parameters ------------------------------------------------

params = get_params(sys.argv[1])
print('Got default params.')

name = params["NAME"]

# ----------------------------------- Set up additional parameters -----------------------------------------------------

# Graph features
in_features = list(params["IN_FEATURES"])
edge_features = list(params["EDGE_FEATURES"])
out_features = list(params["OUT_FEATURES"])

# Survival Information
event_stat = params["EVENT_STAT"]
event_dur = params["EVENT_DUR"]
event_pos = ' '.join(params["POS_EVENT"].split('_'))

# ---------------------------------------------- Loading the Data ------------------------------------------------------

outcomes_df = pd.read_csv(params["SURV_FILE"])
outcomes_df['Virtual_No'] = outcomes_df['Virtual_No'].astype(str)
outcomes_df['core1'] = outcomes_df['core1'].astype(str)
outcomes_df['core2'] = outcomes_df['core2'].astype(str)

tumour100 = outcomes_df[outcomes_df['patients_min_1corewith100'] == 'yes']

node_files, edge_files = order_files(params["NODE_DIR"],
                                     params["EDGE_DIR"])

node_ids = [get_id(f) for f in node_files]

node_files = np.asarray(node_files)
edge_files = np.asarray(edge_files)

if len(node_files) == 0 or len(edge_files) == 0:
    raise Exception(f"Directory empty. Nodes = {len(node_files)}, Edges = {len(edge_files)}")

# Load the train-test split information.
train_test = pd.read_csv(params["TRAIN_TEST_FILE"])
train_test['id'].astype(str)
if train_test['id'].dtype is not str:
    train_test['id'] = train_test['id'].astype(str)

print("Ordered graph data, loaded cross validation file and survival file.")


# ------------------------------------------- Match Survival Data ------------------------------------------------------

time_to_event = []
event_binary = []
core_id = []
for i in tumour100.index:
    if tumour100['core1'].loc[i] != 'nan':
        core_id.append('_'.join((tumour100['Virtual_No'].loc[i], tumour100['core1'].loc[i])))
        event_bin = 1 if tumour100[event_stat].loc[i] == event_pos else 0
        event_binary.append(event_bin)
        tte = tumour100[event_dur].loc[i]
        time_to_event.append(tte)

    if tumour100['core2'].loc[i] != 'nan':
        core_id.append('_'.join((tumour100['Virtual_No'].loc[i], tumour100['core2'].loc[i])))
        event_bin = 1 if tumour100[event_stat].loc[i] == event_pos else 0
        event_binary.append(event_bin)
        tte = tumour100[event_dur].loc[i]
        time_to_event.append(tte)

time_to_event = np.array([time_to_event])
event_binary = np.array([event_binary])
survival = np.concatenate((event_binary, time_to_event), axis=0).T

# The order of the node files and these outputs will be different. So this goes through the
# survival data and finds the index of the survival relative to the node and edge files.
ordered_ids = []
for tt in core_id:
    ordered_ids.append(node_ids.index(tt))

# Now we need to reorder the node and edge files to match the survival data.
node_ids = np.asarray(node_ids)[ordered_ids].tolist()
node_files = node_files[ordered_ids]
edge_files = edge_files[ordered_ids]

print("Matched survival data.")

# ---------------------------------------- Set up metric tracking ------------------------------------------------------

acc_filename = os.path.join(params["METRIC_DIR"], f'{name}_{params["EPOCHS"]}_eval_metrics.txt')
acc_file = open(acc_filename, 'w+')
acc_file.close()

print(f'Created {acc_filename}.')


# --------------------------------------- Get number of folds ----------------------------------------------------------


def get_folds(directory):
    models = [m.split('_')[-2] for m in os.listdir(directory)]
    model_folds = max([int(f.replace('fold', '')) for f in list(set(models))])
    return model_folds


n_folds = get_folds(params["MODEL_DIR"])

# --------------------------------------- Evaluation Fold Loop ---------------------------------------------------------

train_cs = []
val_cs = []
test_cs = []

for fold_i in range(n_folds):
    divider = '-' * 50 + f'Fold {fold_i}' + '-' * 50

    acc_file = open(acc_filename, 'a')
    acc_file.write(divider)
    acc_file.close()

    # --------------------------------------------- Create Graphs ------------------------------------------------------

    train_graphs, train_dur, train_event = [], [], []
    val_graphs, val_dur, val_event = [], [], []
    test_graphs, test_dur, test_event = [], [], []

    for i in range(len(node_files)):
        tissue_id = get_id(node_files[i])
        if tissue_id in train_test['id'].tolist():
            split_set = train_test[train_test['id'] == tissue_id][f'fold_{params["FOLD"]}'].item()
            if split_set == 'train':
                train_dur.append(survival[i][1])
                train_event.append(survival[i][0])

            elif split_set == 'val':
                val_dur.append(survival[i][1])
                val_event.append(survival[i][0])
            else:
                test_dur.append(survival[i][1])
                test_event.append(survival[i][0])

    # Scale the duration down.
    train_dur = np.asarray(train_dur) / 60
    val_dur = np.asarray(val_dur) / 60

    # Clip the values because the output cannot be exactly 0 or 1.
    train_dur = np.clip(train_dur, a_min=1e-7, a_max=(1 - 1e-7))
    val_dur = np.clip(val_dur, a_min=1e-7, a_max=(1 - 1e-7))

    train_surv = [[e, d] for e, d in zip(train_event, train_dur)]
    val_surv = [[e, d] for e, d in zip(val_event, val_dur)]

    n_train_graphs = train_test[f'fold_{params["FOLD"]}'].value_counts()['train']

    counter = 0
    train_counter = 0
    val_counter = 0
    test_counter = 0
    for i in range(len(node_files)):
        tissue_id = get_id(node_files[i])
        if tissue_id in train_test['id'].tolist():
            split_set = train_test[train_test['id'] == tissue_id][f'fold_{params["FOLD"]}'].item()
            if split_set == 'train':
                if counter % 50 == 0:
                    print(f'Loaded [{counter} / {n_train_graphs}]')
                counter += 1

                train_graphs.append(graph_object(node_path=node_files[i],
                                                 edge_path=edge_files[i],
                                                 in_features=in_features,
                                                 edge_features=edge_features,
                                                 sn=params["SN"],
                                                 y=np.expand_dims(train_surv[train_counter], axis=0),
                                                 bin=True,
                                                 ))
                train_counter += 1

            elif split_set == 'val':
                val_graphs.append(graph_object(node_path=node_files[i],
                                               edge_path=edge_files[i],
                                               in_features=in_features,
                                               edge_features=edge_features,
                                               sn=params["SN"],
                                               y=np.expand_dims(train_surv[train_counter], axis=0),
                                               bin=True,
                                               ))
                val_counter += 1

            else:
                test_graphs.append(graph_object(node_path=node_files[i],
                                                edge_path=edge_files[i],
                                                in_features=in_features,
                                                edge_features=edge_features,
                                                sn=params["SN"],
                                                y=np.expand_dims(train_surv[train_counter], axis=0),
                                                bin=True,
                                                ))
                test_counter += 1

    print('Created graphs')

    # ---------------------------------------- Create data loaders -----------------------------------------------------

    censored = np.array([0 if graph.y[0][0] == 0 else 1 for graph in train_graphs])

    Event_idx = np.where(censored == 1)[0]
    Censored_idx = np.where(censored == 0)[0]

    trainLoader = DataLoader(val_graphs, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"])
    valLoader = DataLoader(val_graphs, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"])
    testLoader = DataLoader(val_graphs, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"])

    print(f'\tNo. Train Graphs = {len(train_graphs)}')
    print(f'\tNo. Val Graphs = {len(val_graphs)}')
    print(f'\tNo. Test Graphs = {len(test_graphs)}')

    # ----------------------------------------- Initialise the model ---------------------------------------------------

    model = GraphAttSurv(in_features=len(in_features),
                         out_features=1,
                         out_graph_features=params["HIDDEN_FEATURES"],
                         edge_dims=len(edge_features),
                         attention_features=params["ATT_FEATURES"],
                         n_message_passing=params["N_MESSAGE_PASSING"],
                         batch_size=params["BATCH_SIZE"],
                         )

    # ------------------------------------------ Match model keys ------------------------------------------------------

    model_name = os.path.join(params["MODEL_DIR"], f'{name}_fold{fold_i}_{params["EPOCHS"]}.h5')

    try:
        model.load_state_dict(torch.load(model_name))
        print('Matched model keys successfully.')
    except:
        raise Exception(f"Model does not exist: {model_name}")

    model = model.to(device)

    # --------------------------------------- Evaluate Train Data ------------------------------------------------------

    model.eval()

    train_c = calculate_cindex(model, trainLoader, device)
    val_c = calculate_cindex(model, valLoader, device)
    test_c = calculate_cindex(model, testLoader, device)

    train_cs.append(train_c)
    val_cs.append(val_c)
    test_cs.append(test_c)

    acc_file = open(acc_filename, 'a')
    acc_file.write(f'\nTrain C = {train_c}\n')
    acc_file.write(f'\nVal C = {val_c}\n')
    acc_file.write(f'\nTest C = {test_c}\n')
    acc_file.close()

# ---------------------------------------- Calculating the means -------------------------------------------------------

train_mu, train_se = mean_acc(train_cs)
val_mu, val_se = mean_acc(train_cs)
test_mu, test_se = mean_acc(train_cs)

train_val = f"Mean Train = {train_mu:.5f} +- {train_se:.5f}"
val_val = f"Mean Val = {val_mu:.5f} +- {val_se:.5f}"
test_val = f"Mean Test = {test_mu:.5f} +- {test_se:.5f}"

print(train_val)
print(val_val)
print(test_val)

divider = '-' * 53 + f'Means' + '-' * 53
acc_file = open(acc_filename, 'a')
acc_file.write(divider)
acc_file.write(train_val + '\n')
acc_file.write(val_val + '\n')
acc_file.write(test_val + '\n')
acc_file.close()

print('Evaluation complete.')

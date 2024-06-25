"""
    This script is for training the weakly supervised relative risk prediction algorithm.
    This will create many different directories similar to the node classification script.

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
    │   ├── Metrics

"""

# --------------------------------------------- Install packages -----------------------------------------------------

import os
from os.path import isfile, isdir
import sys
import torch
import torch.optim as optim
import logging
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader

from Utilities.default_args import get_params
from SNGraph.utils import order_files, get_id
from SNGraph.data import graph_object, Sampler_custom
from SNGraph.models import GraphAttSurv
from SNGraph.loss import weighted_mean_squared_error
from SNGraph.eval import cox_loss_sorted, calculate_cindex


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
train_test = pd.read_csv(params["CV_PATH"])
train_test['id'].astype(str)
if train_test['id'].dtype is not str:
    train_test['id'] = train_test['id'].astype(str)

logger.info("Ordered graph data, loaded cross validation file and survival file.")

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

logger.info("Matched survival data.")

# ---------------------------------------- Set up metric tracking ------------------------------------------------------

loss_filename = os.path.join(params["METRIC_DIR"], f'{params["name"]}_loss.txt')
c_filename = os.path.join(params["METRIC_DIR"], f'{params["name"]}_cindex.txt')
if not isfile(loss_filename):
    loss_file = open(loss_filename, 'w+')
    loss_file.close()
    logger.info(f'Created {loss_filename}.')

if not isfile(c_filename):
    c_file = open(c_filename, 'w+')
    c_file.close()
    logger.info(f'Created {c_filename}.')

# --------------------------------------------- Create Graphs ----------------------------------------------------------

train_graphs = []
val_graphs = []
train_dur = []
train_event = []
val_dur = []
val_event = []
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
            continue

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
            continue

logger.info('Created graph objects.')

# ---------------------------------------- Create dataloaders ----------------------------------------------------------

censored = np.array([0 if graph.y[0][0] == 0 else 1 for graph in train_graphs])

Event_idx = np.where(censored == 1)[0]
Censored_idx = np.where(censored == 0)[0]

train_batch_sampler = Sampler_custom(Event_idx, Censored_idx, params["BATCH_SIZE"])
trainLoader = DataLoader(train_graphs, batch_sampler=train_batch_sampler)
valLoader = DataLoader(val_graphs, batch_size=params["BATCH_SIZE"], shuffle=params["SHUFFLE"])

logger.info(f'\tNo. Train Graphs = {len(train_graphs)}')
logger.info(f'\tNo. Val Graphs = {len(val_graphs)}')

# ----------------------------------------- Initialise the model -------------------------------------------------------

model = GraphAttSurv(in_features=len(in_features),
                     out_features=1,
                     out_graph_features=params["HIDDEN_FEATURES"],
                     edge_dims=len(edge_features),
                     attention_features=params["ATT_FEATURES"],
                     n_message_passing=params["N_MESSAGE_PASSING"],
                     batch_size=params["BATCH_SIZE"],
                     )

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

optimizer = optim.SGD(model.parameters(), lr=params["LR"])

# ------------------------------------------------ Train Loop ----------------------------------------------------------

model = model.to(device)

if pre_epochs == 0:
    loss_file = open(loss_filename, 'a')
    loss_file.write(f'\nFold {params["FOLD"]}')
    loss_file.close()

losses = []
train_cs = []
val_cs = []
ave_train_cs = []
ave_test_cs = []
for epoch in range(params["EPOCHS"] + 1):
    epoch_loss = []

    for i, batch in enumerate(trainLoader):
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)
        out, _ = model(batch)

        loss = cox_loss_sorted(out, batch.y[:, 0], batch.y[:, 1])

        # Sometimes dues to an inbalanced dataset, the loss can produce nans. But that's not a problem with the model.
        if loss == loss:
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss.append(loss.item())

        del loss
        del batch
        del out
        torch.cuda.empty_cache()

        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)

    if len(epoch_loss) != 0:
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        losses.append(epoch_loss)
        logging.info(f'\tEpoch [{epoch + 1} / {params["EPOCHS"]}] : loss = {epoch_loss:.7}')
    else:
        logging.info(f'\tEpoch [{epoch + 1} / {params["EPOCHS"]}] : loss = []')

    if epoch % params["SAVE_EPOCHS"] == 0:
        model.eval()

        train_c = calculate_cindex(model, trainLoader, device)
        val_c = calculate_cindex(model, valLoader, device)

        c_string = f'Fold {params["FOLD"]} - Epoch {params["EPOCH"]} - Train C {train_c} - Val C {val_c}'
        logger.info(c_string)

        torch.save(model.state_dict(), f'{model_name}_{pre_epochs + epoch}.h5')
        loss_file = open(loss_filename, 'a')
        loss_file.write('\n')
        loss_file.write('\n'.join(map(str, losses[-params["SAVE_EPOCHS"]:])))
        loss_file.close()

        c_file = open(c_filename, 'a')
        c_file.write(c_string + '\n')
        c_file.close()

        del train_c
        del val_c

        model.train()

logger.info('Complete!')









import numpy as np
import pandas as pd
import torch
import copy
from torch_geometric.data import Data
from torch.utils.data.sampler import Sampler


def one_hot_encoder(data,
                    classes):
    """ A method for taking a list of indices and
    mapping them to a one-hot encoding.

    :param data: List of indices
    :type data: list
    :param classes: List containing all possible labels.
    :type classes: list
    :return:
    """
    one_hot = np.zeros(shape=(len(data), len(classes)))
    for i, ttype in enumerate(data):
        if ttype in classes:
            one_hot[i, classes.index(ttype)] = 1
    return one_hot


def graph_object(node_path,
                 edge_path,
                 node_features,
                 edge_features,
                 out_features=None,
                 out_column=None,
                 sn=0,
                 threshold=None,
                 y=None,
                 bin_out=True,
                 ):
    """
    A method for loading the node and edge csv data for one TMA core into a torch_geometric graph format.

    :param node_path: Path containing nodes and node features.
    :type node_path: str
    :param edge_path: Path containing edges and edge features.
    :type edge_path: str
    :param node_features: A list of node features to be included from all node features.
    :type node_features: list
    :param edge_features: A list of edge features to be included from all edge features.
    :type edge_features: list
    :param out_features: A list of the outputs as they exist in the pandas dataframe
    :type out_features: list
    :param out_column: The column in the dataframe where the outputs exist.
    :type out_column: str
    :param sn: The level of nodes to include when making the graph. This will affect the
            the edges that are created and the nodes that are included. (default :obj:`0`)
    :type sn: int
    :param threshold: If this is specified then only edges below this threshold will be used and
        any edges with longer lengths will be removed. It is recommended to use a size that is
        greater than the supernode connection lengths. (default :obj: `None`)
    :type threshold: float
    :param y: If y is true then output variables will be determined. Otherwise there will
        be no output variables. (default :obj:`True`)
    :type y: bool
    :param bin_out: If bin is true and y is not None then the output y will be a graph level output.
        (default :obj:`True`)
    :param bin_out: bool
    :return: Graph data object.
    """

    node_df = pd.read_csv(node_path) if type(node_path) == str else node_path
    edge_df = pd.read_csv(edge_path) if type(edge_path) == str else edge_path

    if threshold is not None:
        edge_df = edge_df[edge_df['D'] <= threshold]

    node_features_copy = node_features.copy()
    # Create one-hot encodings of the node's supernode status.
    node_df[['SN0', 'SN1', 'SN2']] = one_hot_encoder(node_df['SN'], [0, 1, 2])

    # Here we assume that the edge and node files contain a column containing
    # the node's supernode status as an in the range [0,2].
    # Should be generalised to N supernodes.
    if sn == 0:
        node_df = node_df[node_df['SN'] == 0]
        edge_df = edge_df[edge_df['SN'] == 0]
    elif sn == 1:
        node_df = node_df[node_df['SN'] < 2]
        edge_df = edge_df[edge_df['SN'] < 2]

        # This is to troubleshoot a problem with indexing. If graph has been
        # created correctly then this should not print anything.
        node_max = max(node_df.index.tolist())
        edge_max = max(max(edge_df['source']), max(edge_df['target']))
        if edge_max > node_max:
            print(node_path, f'\nNodes = {node_max}, Edges = {edge_max}')

    # The data need to be in a specific format in the graph.
    input_features = torch.tensor(np.array(node_df[node_features_copy]))
    input_edges = torch.tensor(np.array(edge_df[['source', 'target']]).transpose())
    edge_attrib = torch.tensor(np.exp(-np.array(edge_df[edge_features]) + 1))
    input_features = input_features.type(torch.float32)
    input_edges = input_edges.type(torch.int32)
    edge_attrib = edge_attrib.type(torch.float32)

    # Force the data to be in the correct format.
    input_features = input_features.type(torch.FloatTensor)
    edge_attrib = edge_attrib.type(torch.FloatTensor)

    # Sometimes we don't want to specify an output.
    if y is not None and bin_out:
        target_features = torch.tensor(y)
        target_features = target_features.type(torch.FloatTensor)
        return Data(x=input_features,
                    y=target_features,
                    edge_index=input_edges,
                    edge_attr=edge_attrib)

    elif y is not None and not bin_out:
        target_features = torch.tensor(one_hot_encoder(node_df[out_column], out_features))
        return Data(x=input_features,
                    y=target_features,
                    edge_index=input_edges,
                    edge_attr=edge_attrib)
    else:
        return Data(x=input_features,
                    edge_index=input_edges,
                    edge_attr=edge_attrib)


class Sampler_custom(Sampler):

    def __init__(self, event_list, censor_list, batch_size):
        self.event_list = event_list
        self.censor_list = censor_list
        self.batch_size = batch_size

    def __iter__(self):

        train_batch_sampler = []
        Event_idx = copy.deepcopy(self.event_list)
        Censored_idx = copy.deepcopy(self.censor_list)
        np.random.shuffle(Event_idx)
        np.random.shuffle(Censored_idx)

        Int_event_batch_num = Event_idx.shape[0] // 2
        Int_event_batch_num = Int_event_batch_num * 2
        Event_idx_batch_select = np.random.choice(Event_idx.shape[0], Int_event_batch_num, replace=False)
        Event_idx = Event_idx[Event_idx_batch_select]

        Int_censor_batch_num = Censored_idx.shape[0] // (self.batch_size - 2)
        Int_censor_batch_num = Int_censor_batch_num * (self.batch_size - 2)
        Censored_idx_batch_select = np.random.choice(Censored_idx.shape[0], Int_censor_batch_num, replace=False)
        Censored_idx = Censored_idx[Censored_idx_batch_select]

        Event_idx_selected = np.random.choice(Event_idx, size=(len(Event_idx) // 2, 2), replace=False)
        Censored_idx_selected = np.random.choice(Censored_idx, size=(
            (Censored_idx.shape[0] // (self.batch_size - 2)), (self.batch_size - 2)), replace=False)

        if Event_idx_selected.shape[0] > Censored_idx_selected.shape[0]:
            Event_idx_selected = Event_idx_selected[:Censored_idx_selected.shape[0], :]
        else:
            Censored_idx_selected = Censored_idx_selected[:Event_idx_selected.shape[0], :]

        for c in range(Event_idx_selected.shape[0]):
            train_batch_sampler.append(
                Event_idx_selected[c, :].flatten().tolist() + Censored_idx_selected[c, :].flatten().tolist())

        return iter(train_batch_sampler)

    def __len__(self):
        return len(self.event_list) // 2
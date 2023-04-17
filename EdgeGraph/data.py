import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


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
                 out_features,
                 out_column,
                 sn=0,
                 test=0,
                 threshold=None,
                 y=True,
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
    :param sn: The level of nodes to include when making the graph. This will affect the
            the edges that are created and the nodes that are included. (default :obj:`0`)
    :type sn: int
    :param test: test defines the proportion of the nodes to be tested on in cross validation.
    :type test: float
    :param threshold: If this is specified then only edges below this threshold will be used and
        any edges with longer lengths will be removed. It is recommended to use a size that is
        greater than the supernode connection lengths. (default :obj: `None`)
    :type threshold: float
    :param y: If y is true then output variables will be determined. Otherwise there will
        be no output variables. (default :obj:`True`)
    :type y: bool
    :param out_features: A list of the outputs as they exist in the pandas dataframe
    :type out_features: list
    :param out_column: The column in the dataframe where the outputs exist.
    :type out_column: str
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

    # Define the indices that correspond from the train-test split if training and testing on sub-graphs.
    train_ids, test_ids = train_test_split(node_df, test_size=test)

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
    if y:
        target_features = torch.tensor(one_hot_encoder(node_df[out_column], out_features))
        return Data(x=input_features,
                    y=target_features,
                    edge_index=input_edges,
                    edge_attr=edge_attrib,
                    train_mask=train_ids,
                    test_mask=test_ids)
    else:
        return Data(x=input_features,
                    edge_index=input_edges,
                    edge_attr=edge_attrib,
                    train_mask=train_ids,
                    test_mask=test_ids)


def graph_object_bin_out(node_path,
                         edge_path,
                         node_features,
                         edge_features,
                         sn=0,
                         threshold=None,
                         train_split=None,
                         y=None,
                         ):
    """A method for loading the node and edge csv data for one TMA core

    :param node_path: Path containing nodes and node features.
    :type node_path: str
    :param edge_path: Path containing edges and edge features.
    :type edge_path: str
    :param node_features: A list of node features to be included from all node features.
    :type node_features: list
    :param edge_features: A list of edge features to be included from all edge features.
    :type edge_features: list
    :param sn: The level of nodes to include when making the graph. This will affect the
            the edges that are created and the nodes that are included. (default :obj:`0`)
    :type sn: int
    :param threshold: If this is specified then only edges below this threshold will be used and
        any edges with longer lengths will be removed. It is recommended to use a size that is
        greater than the supernode connection lengths. (default :obj: `None`)
    :type threshold: float
    :param train_split: If train_spit is not None then the nodes will be split into train and test sets.
    :type train_split: float or int
    :param y: If y is not None then provide the binary graph-level output. (default :obj:`None`)
    :type y: float
    :return: Graph data object.
    """

    # Load the node and edge csv data.
    node_df = pd.read_csv(node_path) if type(node_path) == str else node_path
    edge_df = pd.read_csv(edge_path) if type(edge_path) == str else edge_path

    if threshold is not None:
        edge_df = edge_df[edge_df['D'] <= threshold]

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

    # Define the indices that correspond from the train-test split.
    if train_split is not None:
        train_split = train_split if train_split < 1 else (train_split % 100) / 100
        train_ids, test_ids = train_test_split(node_df, test_size=train_split)
    else:
        train_ids, test_ids = train_test_split(node_df)

    # The data need to be in a specific format in the graph.
    input_features = torch.tensor(np.array(node_df[node_features]))
    input_edges = torch.tensor(np.array(edge_df[['source', 'target']]).transpose())
    # edge_attrib = torch.tensor(np.array(edge_df[edge_features]))
    edge_attrib = torch.tensor(np.exp(-np.array(edge_df[edge_features]) + 1))
    input_edges = input_edges.type(torch.int32)
    edge_attrib = edge_attrib.type(torch.float32)

    # Force the data to be in the correct format
    input_features = input_features.type(torch.FloatTensor)
    edge_attrib = edge_attrib.type(torch.FloatTensor)

    # Sometimes we don't want to specify an output.
    if y is None:
        return Data(x=input_features,
                    edge_index=input_edges,
                    edge_attr=edge_attrib,
                    train_mask=train_ids,
                    test_mask=test_ids)
    else:
        target_features = torch.tensor(y)
        target_features = target_features.type(torch.FloatTensor)
        return Data(x=input_features,
                    y=target_features,
                    edge_index=input_edges,
                    edge_attr=edge_attrib,
                    train_mask=train_ids,
                    test_mask=test_ids)
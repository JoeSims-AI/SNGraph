import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

from EdgeGraph.graph import get_node_connections


def plot_nodes_with_labels(node_dataframe,
                           supernode):
    """ This method takes all nodes from the specified level
    and plots them with along with their indices. I used this
    to identify the specific node I want to get the connections
    for when creating visualisations.

    :param node_dataframe: The dataframe containing the xy coordinates.
    :type node_dataframe: pd.DataFrame
    :param supernode: The supernode level to plot.
    :type supernode: positive int > 0
    :return:
    """

    node_dataframe = pd.read_csv(node_dataframe) if type(node_dataframe) == str else node_dataframe
    # Identify the cell nodes and specified supernodes.
    cell_nodes = node_dataframe[node_dataframe['SN'] == 0]
    supernodes = node_dataframe[node_dataframe['SN'] == supernode]

    plt.figure(figsize=(8, 8))
    plt.scatter(cell_nodes['X(px)'],
                cell_nodes['Y(px)'],
                s=0.1,
                color='lightgray')
    plt.scatter(supernodes['X(px)'],
                supernodes['Y(px)'],
                s=20,
                color='tab:orange')

    for i, (x, y) in enumerate(zip(supernodes['X(px)'], supernodes['Y(px)'])):
        plt.text(x, y, supernodes.index.tolist()[i], fontsize=6, horizontalalignment='center', rotation=-30)

    plt.axis('off')
    plt.axis('equal')
    plt.show()


def plot_node_connections_id(node_df,
                             edge_df,
                             idx,
                             l1_size=10,
                             l2_size=20):
    """
    A method for displaying a level1 supernode and its
    connections with the cell nodes.

    :param node_df: A dataframe containing the xy coordinates
        of the cell nodes and supernodes.
    :type node_df: pd.DataFrame
    :param edge_df: A dataframe containing the edges between nodes.
    :type edge_df: pd.DataFrame
    :param idx: The index of the supernode to plot.
    :type idx: int
    :param l1_size: The marker size of the l1 supernodes
    """

    node_df = pd.read_csv(node_df) if type(node_df) == str else node_df
    edge_df = pd.read_csv(edge_df) if type(edge_df) == str else edge_df

    connections = get_node_connections(idx, node_df, edge_df)
    cell_df = node_df[node_df['SN'] == 0]

    plt.figure(figsize=(9, 9))
    plt.scatter(cell_df['X(um)'], cell_df['Y(um)'], s=0.3, color='lightgrey', z_order=0)
    plt.scatter(connections['X(um)'], connections['Y(um)'], s=l1_size, color='tab:blue', zorder=1)
    plt.scatter(node_df['X(um)'].loc[idx], node_df['Y(um)'].loc[idx], s=l2_size, color='tab:red', zorder=2)
    plt.axis('off')
    plt.axis('equal')
    plt.show()

import matplol.pyplot as plt
from matplotlib import cm

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
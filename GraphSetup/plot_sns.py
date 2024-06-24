"""
    This method is to plot the supernodes on top of the cell detections to show that they were created successfully.
    It will plot the the supernodes with labels and you can then input one of the specific supernodes and it will plot
    the connections to the preceding set of nodes.

    python plot_sns.py <param_file_path> <supernode_level>

"""

# --------------------------------------------- Install packages -----------------------------------------------------
import os
from os.path import isdir
import sys

from EdgeGraph.plot import plot_nodes_with_labels, plot_node_connections_id
from Utilities.default_args import get_params

# ------------------------------------------- Load project parameters ------------------------------------------------

params = get_params(sys.argv[1])
sn_level = sys.argv[2]
print('Got default params.')

# ---------------------------------------------- Get and select files --------------------------------------------------

# Get a list of the files
filenames = os.listdir(params[f"SN{sn_level}_DIR"])
for i, f in enumerate(filenames):
    print(f'\t{i}\t\t{f}')

index = int(input('Type the index of the file you want to dispay?\t'))

node_path = os.path.join(params[f'SN{sn_level}_DIR'], "NodeFiles", filenames[index])
edge_path = os.path.join(params[f'SN{sn_level}_DIR'], "EdgeFiles", filenames[index])

plot_nodes_with_labels(node_path,
                       sn_level)

node_id = int(input('What was the index of the node you want to plot?\t'))

plot_node_connections_id(node_path,
                         edge_path,
                         idx=node_id,
                         )





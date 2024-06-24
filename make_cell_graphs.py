"""
    This code takes in a set of parameters, creates a project directory.
    It reads the dataframe containing the locations of the nodes and creates Delaunay graphs based on these.
    The outputs are files containing a dataframe with two columns: 'source' and 'target'.
    These contain the index of the node where the edge begins and the index of the node where the edge ends.

    This code will create 2 subdirectories in the parent directory. This will follow the following format:

    ├── Project
    │   ├── Graphs
    │   │   ├── SN0
    │   │   │   ├── NodeFiles
    │   │   │   │    ├── file0.csv
    │   │   │   ├── EdgeFiles
    │   │   │   │    ├── file0.csv

    Joe 25/04/2024
"""


# --------------------------------------------- Install packages -----------------------------------------------------
import os
from os.path import isdir
import sys

from EdgeGraph.graph import delaunay_edges
from Utilities.default_args import get_params

# ------------------------------------------- Load project parameters ------------------------------------------------

params = get_params(sys.argv[1])
node_dir = sys.argv[2]
print('Got default params.')

# ----------------------------------------------- Setup project ------------------------------------------------------

out_node_dir = os.path.join(params["SN0"], "NodeFiles")
out_edge_dir = os.path.join(params["SN0"], "EdgeFiles")

if isdir(params["SN0"]):
    if not isdir(out_node_dir):
        os.mkdir(out_node_dir)
        print(f'Made directory {out_node_dir}')
    if not os.path.join(out_edge_dir):
        os.mkdir(out_edge_dir)
        print(f'Made directory {out_edge_dir}')

# ----------------------------------------------- Create graphs ------------------------------------------------------

node_filenames = os.listdir(node_dir)
for i, filename in enumerate(node_filenames):

    print(f'Working on [{i+1} / {len(node_filenames)}] - {filename}')

    in_node_path = os.path.join(node_dir, filename)
    out_node_path = os.path.join(out_node_dir, filename)
    out_edge_path = os.path.join(out_edge_dir, filename)

    node_df, edge_df = delaunay_edges(in_node_path,
                                      params["X_COL"],
                                      params["Y_COL"],
                                      params["THRESHOLD"],
                                      True)

    node_df.to_csv(out_node_path, index=False)
    print(f'\tSaved {out_node_path}')
    edge_df.to_csv(out_edge_path, index=False)
    print(f'\tSaved {out_edge_path}')

print('Complete!')

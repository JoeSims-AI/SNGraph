"""
    This method takes the node and edge files from the cell graph and then given a set of arguments, creates the
    first set of supernodes (SNs) connected hierarchically with the cell graph.

    This will introduce a new directory for the hierarchical graphs. So the project will now look like:

    ├── Project
    │   ├── Graphs
    │   │   ├── SN0
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   │   ├── SN1
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   │   │   │    ├── file0.csv
"""

# --------------------------------------------- Install packages -----------------------------------------------------
import os
from os.path import isdir
import sys

from EdgeGraph.supernodes import create_level1_supernodes
from Utilities.default_args import get_params

# ------------------------------------------- Load project parameters ------------------------------------------------

params = get_params(sys.argv[1])
print('Got default params.')

# ----------------------------------------------- Setup project ------------------------------------------------------

out_node_dir = os.path.join(params["SN1"], "NodeFiles")
out_edge_dir = os.path.join(params["SN1"], "EdgeFiles")

if not isdir(params["SN1"]):
    os.mkdir(params["SN1"])
    print(f'Made directory {params["SN1"]}')
if not isdir(out_node_dir):
    os.mkdir(out_node_dir)
    print(f'Made directory {out_node_dir}')
if not os.path.join(out_edge_dir):
    os.mkdir(out_edge_dir)
    print(f'Made directory {out_edge_dir}')

# ----------------------------------------------- Create graphs ------------------------------------------------------

node_filenames = os.listdir(os.path.join(params["SN0"], "NodeFiles"))
completed = os.listdir(out_node_dir)
uncompleted = [f for f in node_filenames if f not in completed]

for i, filename in enumerate(uncompleted):

    print(f'Working on [{i + 1} / {len(uncompleted)}] - {filename}')

    in_node_path = os.path.join(params["SN0"], "NodeFiles", filename)
    in_edge_path = os.path.join(params["SN0"], "EdgeFiles", filename)
    out_node_path = os.path.join(params["SN1"], "EdgeFiles", filename)
    out_edge_path = os.path.join(params["SN1"], "EdgeFiles", filename)

    node_df, edge_df = create_level1_supernodes(in_node_path,
                                                in_edge_path,
                                                sep=params["SEPARATION"],
                                                grid_type=params["LAYOUT"],
                                                radius_f=params["RADIUS"],
                                                save=False
                                                )
    node_df.to_csv(out_node_path, index=False)
    print(f'\tSaved {out_node_path}')
    edge_df.to_csv(out_edge_path, index=False)
    print(f'\tSaved {out_edge_path}')

print('Complete!')







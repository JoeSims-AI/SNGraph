"""
    This method takes the node and edge files from the cell graph and then given a set of arguments, creates the
    first set of supernodes (SNs) connected hierarchically with the cell graph.

    This code is an example of creating a maximum of 2 sets of supernodes but could be used indefinitely to create as
    many sequential layers as you could ever want depending on what parameters you specify.

    python make_sn2s.py <param_file_location>

    This will introduce a new directory for the hierarchical graphs. So the project will now look like:

    ├── Project
    │   ├── Graphs
    │   │   ├── SN0
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   │   ├── SN1
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   │   ├── SN2
    │   │   │   ├── NodeFiles
    │   │   │   ├── EdgeFiles
    │   │   │   │    ├── file0.csv
"""

# --------------------------------------------- Install packages -----------------------------------------------------
import os
from os.path import isdir
import sys

from SNGraph.supernodes import create_supernode_level
from Utilities.default_args import get_params

# ------------------------------------------- Load project parameters ------------------------------------------------

params = get_params(sys.argv[1])
print('Got default params.')

# ----------------------------------------------- Setup project ------------------------------------------------------

out_node_dir = os.path.join(params["SN2"], "NodeFiles")
out_edge_dir = os.path.join(params["SN2"], "EdgeFiles")

if not isdir(params["SN2"]):
    os.mkdir(params["SN2"])
    print(f'Made directory {params["SN2"]}')
if not isdir(out_node_dir):
    os.mkdir(out_node_dir)
    print(f'Made directory {out_node_dir}')
if not os.path.join(out_edge_dir):
    os.mkdir(out_edge_dir)
    print(f'Made directory {out_edge_dir}')

# ----------------------------------------------- Create graphs ------------------------------------------------------

node_filenames = os.listdir(os.path.join(params["SN1"], "NodeFiles"))
completed = os.listdir(out_node_dir)
uncompleted = [f for f in node_filenames if f not in completed]


for i, filename in enumerate(uncompleted):

    print(f'Working on [{i + 1} / {len(uncompleted)}] - {filename}')

    in_node_path = os.path.join(params["SN0"], "NodeFiles", filename)
    in_edge_path = os.path.join(params["SN0"], "EdgeFiles", filename)
    out_node_path = os.path.join(params["SN1"], "EdgeFiles", filename)
    out_edge_path = os.path.join(params["SN1"], "EdgeFiles", filename)

    node_df, edge_df = create_supernode_level(in_node_path,
                                              in_edge_path,
                                              separation=params["SN2_SEPARATION"],
                                              radius_f=params["SN_RADIUS"],
                                              supernode=2
                                              )

    node_df.to_csv(out_node_path, index=False)
    print(f'\tSaved {out_node_path}')
    edge_df.to_csv(out_edge_path, index=False)
    print(f'\tSaved {out_edge_path}')

print('Complete!')
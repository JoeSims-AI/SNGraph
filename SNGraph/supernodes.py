from tqdm import tqdm
import numpy as np
import shapely.geometry as g
from SNGraph.graph import get_core_outlines
from SNGraph.utils import *


def square_grid(x_range,
                y_range,
                sep):
    """
    Given an x and y range along with an edge length, this will return a set of coordinates that lay on the vertices
    of a square grid in this range.
    :param x_range: The minimum and maximum x coordinates.
    :type x_range: list
    :param y_range: The minimum and maximum x coordinates.
    :type y_range: range
    :param sep: The edge length.
    :type sep: int
    :return: vertex coordinates.
    """

    # Define the number of horizontal supernodes.
    h_dots = int(abs((x_range[0] - x_range[1])) // sep)
    # Define the number of vertical supernodes.
    v_dots = int(abs((y_range[0] - y_range[1])) // sep)

    # A list of the supernode coordinates.
    locs = []
    # We want to include the limit so we iterate to the number of horizontal supernodes +1
    for h in range(h_dots + 1):
        for v in range(v_dots + 1):
            # The coordinates are the bottom left corner coordinates + (the separation * index of supernode)
            locs.append([(h * sep) + min(x_range), (v * sep) + min(y_range)])
    return np.asarray(locs)


def tri_grid(x_range,
             y_range,
             sep):
    """
    Given an x and y range along with an edge length, this will return a set of coordinates that lay on the vertices
    of an equilateral triangular grid in this range.
    :param x_range: The minimum and maximum x coordinates.
    :type x_range: list
    :param y_range: The minimum and maximum x coordinates.
    :type y_range: range
    :param sep: The edge length.
    :type sep: int
    :return: vertex coordinates.
    """

    # Define the number of horizontal supernodes.
    h_dots = int(abs((x_range[0] - x_range[1])) // sep)
    # Define the number of vertical supernodes.
    v_dots = int(abs((y_range[0] - y_range[1])) // (0.866*sep))

    # A list of the supernode coordinates.
    locs = []
    # We want to include the limit so we iterate to the number of horizontal supernodes +1
    for h in range(h_dots + 1):
        for v in range(v_dots + 1):
            if v % 2 == 0:
                locs.append([min(x_range) + (h * sep), min(y_range) + (0.866 * v * sep)])
            else:
                locs.append([min(x_range) + (sep * (0.5 + h)), min(y_range) + (0.866 * v * sep)])
    return np.asarray(locs)


def create_level1_supernodes(node_path,
                             edge_path,
                             outline=None,
                             sep=200,
                             grid_type='square',
                             radius_f=1,
                             ):
    """ This method takes the cell graph and then creates regularly spaced nodes
    that lay on the vertices of a square grid of with side length specified by the
    'separation' argument.
    There are a lot of scenarios to account for so this code is a little messy.
    Also the separation is specified in um.

    :param node_path: The path to the csv file containing the nodes.
    :type node_path: str
    :param edge_path: The path to the edge file containing the connections.
    :type edge_path: str
    :param sep: The distance between adjacent supernodes in um. (default :obj:`200`)
    :type sep: positive int
    :param outline: If there is an xml outline for the file for the tissue then use that. If False, then a square
        outline will be produced using the maxima and minima of the cell locations. (default :obj:`True`)
    :type outline: None
    :param grid_type: If you want the supernodes to be laid out in a square grid or an equilateral triangular grid.
        ('square', 'tri'). (default :obj:`square`)
    :type grid_type: str
    :return:
    """

    # Import csv containing nodes and node features.
    nodes = pd.read_csv(node_path)
    # Import csv containing edges and edge features.
    edges = pd.read_csv(edge_path)

    x_sf, y_sf = calculate_sf(nodes,
                              numerator='px',
                              denominator='um')

    if outline is not None:
        outline = outline

    else:
        tissue_id = get_id(node_path)
        xs = nodes[f'X(um)'].min()
        xe = nodes[f'X(um)'].max()
        ys = nodes[f'Y(um)'].min()
        ye = nodes[f'Y(um)'].max()
        outline = g.Polygon(np.asarray([[xs, ys],
                                        [xs, ye],
                                        [xe, ye],
                                        [xe, ys]]))

    # We want to define a box around the tissue to create the supernodes in.
    # So the minimum and maximum supernode positions will be at these point.
    x_min = round_to(min(outline.exterior.xy[0]), int(2*sep))
    x_max = round_to(max(outline.exterior.xy[0]), int(2*sep))
    y_min = round_to(min(outline.exterior.xy[1]), int(2*sep))
    y_max = round_to(max(outline.exterior.xy[1]), int(2*sep))

    # Define the number of horizontal supernodes.
    h_dots = int((x_max - x_min) // sep)
    # Define the number of vertical supernodes.
    v_dots = int((y_max - y_min) // sep)

    # A list of the supernode coordinates.
    if grid_type == 'square':
        sn1_locs = square_grid([x_min, x_max], [y_min, y_max], sep)
    else:
        sn1_locs = tri_grid([x_min, x_max], [y_min, y_max], sep)

    # This list will collect the supernodes that lie within the core outline.
    keep_id = []
    for i in range(len(sn1_locs)):
        if outline.contains(g.Point(sn1_locs[i])):
            keep_id.append(i)
    sn1_locs_kept = sn1_locs[keep_id]

    # Now we have the locations of the supernodes and now we need to form the connections with the cell nodes.
    # When we make these connections we want calculate the edge features as well.

    time1 = time.time()
    # Collect the index of the supernode and the cell it's connected to.
    pairs = []
    # Collect the distances between the supernode and cell.
    dists = []
    # Get the x and y distances between the supernode and cell.
    dxdys = []
    keep_keep = []
    counter = 0
    # Iterate through length of the total kept supernodes.
    for i, sn in tqdm(enumerate(sn1_locs_kept)):
        # Collect the cells that are within a square box centered on the supernode
        # with width and height  = 2* separation
        x, y = sn[0], sn[1]
        # Confine the surrounding cells within a given radius.
        cond = (nodes['X(um)'] <= x+(sep * radius_f)) & (nodes['X(um)'] >= x-(sep * radius_f)) & \
               (nodes['Y(um)'] <= y+(sep * radius_f)) & (nodes['Y(um)'] >= y-(sep * radius_f))

        sn1s_df = nodes[cond][['X(um)', 'Y(um)']]
        sn1s = np.asarray(sn1s_df)

        dist = np.linalg.norm(sn - sn1s, axis=1, keepdims=False)

        dxdy = np.absolute(np.subtract(sn, sn1s))
        pair_list = np.array([[counter + len(nodes)] * len(sn1s_df), sn1s_df.index.tolist()]).T
        dist_cond = (dist <= (sep*radius_f))
        if dist_cond.sum() > 0:
            dists.extend(dist[dist_cond].tolist())
            dxdys.extend(dxdy[dist_cond].tolist())
            pairs.extend(pair_list[dist_cond].tolist())
            keep_keep.append(i)
            counter += 1

    sn1_locs_kept = sn1_locs_kept[keep_keep]
    print(f'\tRemoved outliers ({time.time() - time1 :.5f} s)')
    # Make sure that the pairs are all integers
    pairs = np.asarray(pairs, dtype=np.int32)
    # Convert the distances to numpy arrays
    dxdys = np.asarray(dxdys, dtype=np.float32)
    # Convert the Euc distances to numpy arrays
    dists = np.asarray(dists, dtype=np.float32)
    # Since the shape of the dists is (n,) we need to make it (n,1) to be able to do things with it.
    dists = dists.reshape((len(dists), 1))
    # Put all of these for all supernodes edge pairs into a dataframe.
    sn1_edges = pd.DataFrame({'source': pairs[:, 0],
                              'target': pairs[:, 1],
                              'dx': dxdys[:, 0],
                              'dy': dxdys[:, 1],
                              'D': dists[:, 0]})

    time2 = time.time()
    # So we have the connections with the cell nodes. Now we want to calculate the mean for
    # each of the connections and store these as the features for the supernodes.
    # Create a new dataframe for the data
    sn_df = pd.DataFrame(columns=nodes.columns)
    # For each of the unique supernodes collect all pairs for that supernode
    for sn in sn1_edges['source'].unique().tolist():
        temp_df = sn1_edges[sn1_edges['source'] == sn]
        # First get the the x and y coordinates of the supernode.
        sn_loc = sn1_locs_kept[sn - len(nodes)]
        if len(temp_df) == 1 and temp_df['target'].iloc[0] == sn:
            # These attributes have to be specified.
            attr = {'X(um)': sn_loc[0],
                    'Y(um)': sn_loc[1],
                    'X(px)': sn_loc[0] * x_sf,
                    'Y(px)': sn_loc[1] * y_sf,
                    'class': 'notype',
                    'Type': 0}
            # For the attributes that don't need to be specified we can just calculate the mean
            for column in sn_df.columns:
                if column not in attr:
                    attr[column] = 0
        else:
            # Get all data for these specific targets corresponding to this supernode.
            temp_df = nodes.loc[temp_df['target']]
            # These attributes have to be specified.
            attr = {'X(um)': sn_loc[0],
                    'Y(um)': sn_loc[1],
                    'X(px)': sn_loc[0] * x_sf,
                    'Y(px)': sn_loc[1] * y_sf,
                    'class': 'notype',
                    'Type': temp_df['Type'].value_counts().keys()[0]}
            # For the attributes that don't need to be specified we can just calculate the mean
            for column in sn_df.columns:
                if column not in attr:
                    attr[column] = np.mean(temp_df[column])
        # Convert this data to a dataframe.
        sn_df = sn_df.append(attr, ignore_index=True)
    # Change the indices to iterate from the max of the cell detections to the number of supernodes.
    sn_df.index = range(len(nodes), len(nodes) + len(sn_df))
    print(f'\tGot supernode connections in ({time.time() - time2 :.5f} s)')

    # We now define which nodes are supernodes and which are cells. This is so we can easily filter them later.
    sn_df['SN'] = [1] * len(sn_df)
    sn1_edges['SN'] = [1] * len(sn1_edges)
    # sn_edges['SN'] = [1] * len(sn_edges)
    edges['SN'] = [0] * len(edges)
    nodes['SN'] = [0] * len(nodes)

    # Concatenate the cell and supernode data.
    total_nodes = pd.concat([nodes, sn_df])
    # concatenate cell edges and supernode connections.
    # total_edges = pd.concat([edges, sn1_edges, sn_edges])
    total_edges = pd.concat([edges, sn1_edges])
    # The indices for the detections was previously specified so only need to change edge file.
    total_edges.index = range(len(total_edges))

    return total_nodes, total_edges


def create_supernode_level(node_path,
                           edge_path,
                           separation=2,
                           radius_f=1,
                           supernode=2):
    """ Creating the first level of supernodes has it's own limitations with it being
    connected to the cell graph so I created a separate method for creating supernodes
    beyond the first level. This method should work for levels > 1 but is yet to be
    tested above 1 level of supernodes. Because I didn't have enough time to dedicate to
    combining the level 1 conditions with these, we have to methods.
    As with the L1 method above, this works in 'um'.
    For this specific method, the subsequent set of supernodes have the coordinates of every other
    supernode from the previous set (like a the same grid but of twice the size.)


    :param node_path: The path to the csv file containing the nodes.
    :type node_path: str
    :param edge_path: The path to the csv file containing the edges.
    :type edge_path: str
    :param outline: This determines whether to obtain the tma and tissue number (outline=True)
        or the id (outline=false). (default :obj:`True`).
    :type outline: bool
    :param separation: I couldn't think of a better name. If you want this layer of supernodes to lie
        on top over every 2nd node then this will be 2. If it is every 3rd previous-layer supernode then
        it will be 3 and so on.
    :type separation: int
    :param radius_f: This is the connection radius as a factor of supernode separation. For example if you want the
        connection radius to be equal to the separation of each supernode then radius_f=1. (default :obj:`1`)
    :param supernode: The hierarchical position of the supernodes that are to be created. (default :obj:`2`).
    :type supernode: positive int >= 2
    :return:
    """

    # Import the node data containing the previous supernode data.
    node = pd.read_csv(node_path)
    # Import edge data containing the previous supernode data.
    edge = pd.read_csv(edge_path)

    # Get the scale factors between the pixels and um positions.
    x_sf = np.mean(node['X(px)']) / np.mean(node['X(um)'])
    y_sf = np.mean(node['Y(px)']) / np.mean(node['Y(um)'])

    # Due to bad labelling, I've labelled the previous supernodes as sn1s but it
    # should really be sn_previous
    sn1s = node[node['SN'] == supernode - 1]
    # The sns are a regular grid and so will share some x and y coordinates.
    x_set = sorted(list(set(sn1s['X(um)'])))
    y_set = sorted(list(set(sn1s['Y(um)'])))

    # This is going to collect the coordinates for every other sn
    # These will be the coordinates of the next layer of supernodes.
    new_sn_coords = []
    for i in range(len(x_set)):
        for j in range(len(y_set)):
            if i % separation == 0 and j % separation == 0:
                new_sn_coords.append([x_set[i], y_set[j]])
    new_sn_coords = np.asarray(new_sn_coords, dtype=np.int32)
    old_sn_coords = np.asarray(sn1s[['X(um)', 'Y(um)']], dtype=np.int32)

    # This is the separation of the previous sn coordinates.
    node_separation = x_set[1] - x_set[0]

    # Converts the coordinates to a list of tuples.
    old_sn_coords = [tuple(pair) for pair in old_sn_coords]
    new_sn_coords = [tuple(pair) for pair in new_sn_coords]

    # Some of the coordinates will lay outside of the tissue
    # So only keep coordinates of new sns that lay on old sns.
    keep_ids = []
    for i, pair in enumerate(new_sn_coords):
        if pair in old_sn_coords:
            keep_ids.append(i)
    new_sn_coords = np.asarray(new_sn_coords, dtype=np.int32)
    keep_new_sns = new_sn_coords[keep_ids]

    pairs = []
    dists = []
    dxdys = []
    # Get the distance between each new sn and the previous layer sns that are 1 step away.
    # Then appending the indices of these pairs, along with the distances.
    sn1s_coords = np.asarray(sn1s[['X(um)', 'Y(um)']])
    for i, sn1 in tqdm(enumerate(keep_new_sns)):

        dist = np.linalg.norm(sn1 - sn1s_coords, axis=1, keepdims=False)
        dxdy = np.absolute(np.subtract(sn1, sn1s_coords))
        pair_list = np.array([[i + len(node)] * len(sn1s), sn1s.index.tolist()]).T
        cond = ((node_separation * (separation-1) * radius_f) + 10 > dist)
        pairs.extend(pair_list[cond].tolist())
        dists.extend(dist[cond].tolist())
        dxdys.extend(dxdy[cond].tolist())

    pairs = np.asarray(pairs, dtype=np.int32)
    dxdys = np.asarray(dxdys, dtype=np.float32)
    dists = np.asarray(dists, dtype=np.float32)
    dists = dists.reshape((len(dists), 1))

    sn2_edges = pd.DataFrame({'source': pairs[:, 0],
                              'target': pairs[:, 1],
                              'dx': dxdys[:, 0],
                              'dy': dxdys[:, 1],
                              'D': dists[:, 0]})

    # Create a new column to show which layer of sns these are.
    sn2_edges['SN'] = [supernode] * len(sn2_edges)

    total_edges = pd.concat([edge, sn2_edges])
    total_edges.index = range(len(total_edges))

    sn_df = pd.DataFrame(columns=node.columns)
    # For each new sn, it gets all of the nodes that it's connected to
    # Then aggregates their information using the mean. This creates a new row for that sn.
    for sn in sn2_edges['source'].unique().tolist():
        temp_df = sn2_edges[sn2_edges['source'] == sn]
        if len(temp_df) == 1 and temp_df['target'].iloc[0] == sn:
            sn_loc = keep_new_sns[sn - len(node)]
            attr = {'X(um)': sn_loc[0],
                    'Y(um)': sn_loc[1],
                    'X(px)': sn_loc[0] * x_sf,
                    'Y(px)': sn_loc[1] * y_sf,
                    'class': 'notype',
                    'Type': 0}

            for column in sn_df.columns:
                if column not in attr:
                    attr[column] = 0

        else:
            temp_df = node.loc[temp_df['target']]

            sn_loc = keep_new_sns[sn - len(node)]
            attr = {'X(um)': sn_loc[0],
                    'Y(um)': sn_loc[1],
                    'X(px)': sn_loc[0] * x_sf,
                    'Y(px)': sn_loc[1] * y_sf,
                    'class': 'notype',
                    'Type': temp_df['Type'].value_counts().keys()[0]}

            for column in sn_df.columns:
                if column not in attr:
                    attr[column] = np.mean(temp_df[column])

        sn_df = sn_df.append(attr, ignore_index=True)

    sn_df['SN'] = [supernode] * len(sn_df)
    sn_df.index = range(len(node), len(node) + len(sn_df))
    total_nodes = pd.concat([node, sn_df])

    return total_nodes, total_edges
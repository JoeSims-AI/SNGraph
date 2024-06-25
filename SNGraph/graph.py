import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import xml.etree.ElementTree as ET
import shapely.geometry as g
from scipy import interpolate
from scipy.spatial import Delaunay
import itertools
from SNGraph.utils import *
from collections import Counter


def get_polygon(coordinates,
                xsf=1,
                ysf=1,
                samples=100,
                convert=True):
    """This method takes the xy coordinates that make up the vertices
    of polygon. It then interpolates them using a b-spline and samples
    from the spline to create more a polygon with more vertices to make
    the polygon appear smoother. The main purpose of this method in
    context of tissue micro-arrays which are circular, was to get a
    better estimation of the area and to not underestimate the area.
    The returned object is a shapely polygon object.
    :param coordinates: The coordinates of the polygon vertices.
    :type coordinates: np.array
    :param xsf: A scale factor to linearly transform the x coordinates.
    :type xsf: float
    :param ysf: A scale factor to linearly transform the y coordinates.
    :type ysf: float
    :param samples: How many points to sample from the spline between
        existing vertices
    :type samples: int
    :param convert: If convert then convert to a polygon, else return the coordinates.
    :return:
    """

    ctr = np.array(coordinates)

    x = ctr[:, 0]
    y = ctr[:, 1]

    if x[-1] != x[0] and y[-1] != y[0]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    try:
        tck, u = interpolate.splprep([x, y], k=3, s=0)
    except:
        points = []
        for i in range(len(coordinates)):
            # The coordinates are converted from px to um.
            points.append(g.Point(coordinates[i][0] / xsf, coordinates[i][1] / ysf))

    else:
        u = np.linspace(0, 1, num=samples * len(x), endpoint=True)
        out = interpolate.splev(u, tck)

        points = []
        for i in range(len(out[0])):
            # The coordinates are converted from px to um.
            points.append(g.Point(out[0][i] / xsf, out[1][i] / ysf))

    return g.Polygon(points) if convert else points


def get_core_outlines(file_path,
                      xsf,
                      ysf):
    """This method opens xml files containing the xy
    coordinates of the tissue micro-array (TMA) core
    outline vertices. It returns a list of the core
    outlines as polygons and a list of the corresponding
    tissue names.

    :param file_path: The file path of the xml.
    :type file_path: str
    :param xsf: The scale factor to linearly transform the x coordinates.
    :type xsf: float
    :param ysf: The scale factor to linearly transform the y coordinates.
    :type ysf: float
    :return: list of polygons for each core outline.
    """

    tree = ET.parse(file_path)
    root = tree.getroot()
    outlines = []
    tissue_names = []

    for annotation in root:
        for curve in annotation:
            # Some of the xml files contain ellipse which are not
            # outlines and so should be ignored.
            if 'ellipse' not in curve.attrib['name'].lower():
                tissue_names.append(curve.attrib['name'])
                xy_coordinates = []
                for vertex in curve:
                    xy_coordinates.append((float(vertex.attrib['x']),
                                           float(vertex.attrib['y'])))

                outlines.append(get_polygon(xy_coordinates, xsf, ysf))

    return outlines, tissue_names


def remove_outlier_detections(file_path,
                              outlines_path,
                              tma_id,
                              out_dir):
    """This method is to take the cell detections from the entire
    tissue micro-array (TMA), retrieve all core outlines and then
    return csv files for each individual TMA core with outliers removed.
    The outliers can occur when the cell detector picks up on objects
    in the background/outside of the core annotations.

    :param file_path: The csv file path containing all cell detections within the tma.
    :type file_path: str
    :param outlines_path: This contains the path to the xml file containing all
        core outlines in the TMA.
    :type outlines_path: str
    :param tma_id: A 6 digit id.
    :type tma_id: str
    :param out_dir: The directory to save the detections for the individual cores.
    :param out_dir: str
    :return:
    """

    # Load the csv file containing all cell detections in one tma
    detections = pd.read_csv(file_path) if type(file_path) == str else file_path

    # Get the scale factor between px and um because xml are in px.
    xsf, ysf = calculate_sf(detections, 'px', 'um')
    polygons, names = get_core_outlines(outlines_path, xsf, ysf)

    # Go through each outline.
    for outline, name in zip(polygons, names):
        # We want a faster way of checking if all cells in the TMA lie
        # within a specific core, so we get the outline coordinates
        # constrain the cell detections within the extremities of the outline.
        # Then we check if those cells lie within the outline or not.
        outline_coordinates = outline.exterior.xy
        xmax, xmin = max(outline_coordinates[0]), min(outline_coordinates[0])
        ymax, ymin = max(outline_coordinates[1]), min(outline_coordinates[1])

        constrained = detections[detections['X(um)'] <= xmax]
        constrained = constrained[constrained['X(um)'] >= xmax]
        constrained = constrained[constrained['Y(um)'] <= ymax]
        constrained = constrained[constrained['Y(um)'] >= ymin]
        constrained.index = range(len(constrained))

        # Now we have the detections contained to a box of a similar size
        # to the core outline, we can go through each detection and find the
        # outliers and remove them.
        in_detection_ids = []
        for i in range(len(constrained)):
            if outline.contains(g.Point(constrained['X(um)'].iloc[i],
                                        constrained['Y(um)'].iloc[i])):
                in_detection_ids.append(i)

        # Now we have identified the ids of the detections within the core
        # we can select them cell detection dataframe.
        if len(constrained) != 0 and len(in_detection_ids) != 0:
            constrained = constrained.loc[in_detection_ids]
            constrained.index = range(len(constrained))
            constrained.to_csv(out_dir + f"{tma_id}_{name}_{get_formatted_date()}.csv", index=False)
        else:
            print(f"This tissue ({tma_id} : {name}) didn't have any detections")


def get_annotation_outlines(file_path,
                            xsf,
                            ysf):
    """
    We already have one method that is specific to the format of retrieving the
    outlines for the TMA cores. This method is for retrieving the outlines for normal
    annotations.

    :param file_path: The path to the xml file containing the outline vertices.
    :type file_path: str
    :param xsf: The linear scale factor to transform the x pixel coordinates to um.
    :type xsf: float
    :param ysf: The linear scale factor to transform the y pixel coordinates to um.
    :type ysf: float
    :return:
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    for annotation in root:
        outlines = []
        for curve in annotation:
            xy_coordinates = []
            for vertex in curve:
                xy_coordinates.append((float(vertex.attrib['x']),
                                       float(vertex.attrib['y'])))
            outlines.append(get_polygon(xy_coordinates, xsf, ysf))
        return outlines


def create_kdtree_edges(file_path,
                        output_dir,
                        leaf_size=3,
                        k=5,
                        ):
    """
    This method takes the locations of a set of nodes and applies a kd-tree
    to determine the edge connections. In this method, edge features are also
    calculated. There are 3 edge features node1(x) - node2(x), node1(y) - node2(y),
    and |node1 - node2|.

    :param file_path: The file path to the csv containing the node xy coordinates
        of the graph nodes.
    :type file_path: str
    :param output_dir: The directory that you would like to save the output graph to.
    :type output_dir: str
    :param leaf_size: Number of point at which to change to a brute force method.
        (default :obj:`3`)
    :type leaf_size: positive int
    :param k: Number of nearest neighbours/connections to make. (default :obj:`5)
    :type k: positive int
    :return:
    """

    coords = np.array(pd.read_csv(file_path)[['X(um)', 'Y(um)']])
    tree = KDTree(coords, leaf_size=leaf_size)
    dist, ind = tree.query(coords, k=k)

    edges = np.zeros((coords.shape[0] * 4, 5))
    for i in range(1, k):
        edges[coords.shape[0] * (i - 1):coords.shape[0] * i, 0] = ind[:, 0]
        edges[coords.shape[0] * (i - 1):coords.shape[0] * i, 1] = ind[:, i]
        edges[coords.shape[0] * (i - 1):coords.shape[0] * i, 2] = dist[:, i]
        edges[coords.shape[0] * (i - 1):coords.shape[0] * i, 3:5] = coords[ind[:, 0]] - coords[ind[:, i]]

    edges_df = pd.DataFrame(data=edges, columns=['source', 'target', 'D', 'dx', 'dy'])
    edges_df['source'] = edges_df['source'].apply(np.int32)
    edges_df['target'] = edges_df['target'].apply(np.int32)

    date = get_formatted_date()
    identity = get_id(file_path)
    output_path = f"{output_dir}{identity}_{date}.csv"
    edges_df.to_csv(output_path, index=False)
    print(f'Saved {identity}. Shape {edges_df.shape}')


def delaunay_edges(file_path,
                   x_col,
                   y_col,
                   return_node_df=True
                   ):
    """
    This method uses Delaunay triangulation to form the graph edges.
    It also returns the edge lengths as the edge features.
    :param file_path: This can be a path to the csv file or a pandas dataframe.
    :type file_path: str
    :param x_col: This the name of the column containing the x coordinates of the cell nodes.
    :param y_col:This the name of the column containing the y coordinates of the cell nodes.
    :param return_node_df:
    :return: Edges
    """
    nodes_df = pd.read_csv(file_path)
    coords = np.asarray(nodes_df[[x_col, y_col]])

    # First get the Delaunay triangles.
    tri = Delaunay(coords)
    tris = tri.simplices

    # Now we want to create a list of all possible pairs and remove any repeating ones.
    pairs = np.concatenate((tris[:, [0, 1]], tris[:, [0, 2]], tris[:, [1, 2]]))
    pairs = np.sort(pairs, axis=1)
    pairs = np.unique(pairs, axis=0)

    edge_lengths = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    dxdys = np.abs(coords[pairs[:, 0]] - coords[pairs[:, 1]])

    edges_df = pd.DataFrame({'source': pairs[:, 0],
                             'target': pairs[:, 1],
                             'D': edge_lengths,
                             'dx': dxdys[:, 0],
                             'dy': dxdys[:, 1]
                             })
    if return_node_df:
        return nodes_df, edges_df
    else:
        return edges_df


def delaunay_with_threshold(delaunay_tri,
                            coords,
                            threshold):
    """
    This method removes any triangles with edges longer than the threshold. If no threshold is
    specified then it will just return the Delaunay triangles.
    :param delaunay_tri:
    :param coords: The x and y coordinates of the nodes
    :param threshold: The threshold to determine the triangles to keep. (default :obj `None`)
    :type threshold: float/int
    :return:
    """

    if threshold is not None:
        # Get the vertices (node indices).
        verts = delaunay_tri.vertices
        # Determine a mask for the triangles containing edges less than the threshold.
        mask = np.zeros_like(verts)
        # Calculate the length of each edge in the triangle and then store it in mask.
        mask[:, 0] = np.linalg.norm(coords[verts[:, 0]] - coords[verts[:, 1]], axis=1)
        mask[:, 1] = np.linalg.norm(coords[verts[:, 0]] - coords[verts[:, 2]], axis=1)
        mask[:, 2] = np.linalg.norm(coords[verts[:, 1]] - coords[verts[:, 2]], axis=1)
        # If a length threshold is specified then convert the mask to boolean.
        mask = mask < threshold
        # Below only returns true if all 3 entries are below the threshold (False).
        mask = mask.all(axis=1)
        return delaunay_tri.simplices[mask, :]
    else:
        return delaunay_tri.simplices


def get_delaunay_from_edges(node_file,
                            edge_file,
                            threshold=None):
    """
    This method take the node and edge file and creates the delaunay triangulation from these edges. This method allows
    disconnected regions to stay disconnected.
    :param node_file: The file of pandas dataframe that contains the node information including the coordinates of the
        between the cells.
    :type node_file: str or pd.DataFrame
    :param edge_file: The file of pandas dataframe that contains the edge information including the Euclidean distance
        between indices.
    :type edge_file: str or pd.DataFrame
    :param threshold: This is a distance threshold. Any edge with a distance shorter than this will be removed.
        (default :obj: `None`)
    :type threshold: int or float
    :return simplices: This contains the indices for each Delaunay triangle formed.
    """

    # Load data
    edge_file = pd.read_csv(edge_file) if type(edge_file) == str else edge_file
    node_file = pd.read_csv(node_file) if type(node_file) == str else node_file

    # First remove the supernodes.
    edge_file = edge_file[edge_file['SN'] == 0]

    # Create a copy of edge file in numpy array format.
    if threshold is not None:
        edge_file = edge_file[edge_file['D'] <= threshold]

    edges_np = np.asarray(edge_file[['source', 'target']])

    # Create a copy of the edge file but with source and target flipped.
    edge_file_copy = edge_file.copy()
    edge_file_copy['source'] = edge_file['target']
    edge_file_copy['target'] = edge_file['source']
    edge_file = pd.concat([edge_file, edge_file_copy], axis=0)

    simps = []
    tuple_edges = [tuple(sorted(e)) for e in edges_np]
    for i in range(edges_np.max()):
        if (i+1) % 100 == 0:
            print(f'[{i + 1} / {edges_np.max()}]')

        temp_df = edge_file[edge_file['source'] == i]
        combs = [tuple(sorted(c)) for c in list(itertools.combinations(temp_df['target'].tolist(), 2))]
        for c in combs:
            if c in tuple_edges:
                simps.append([i, c[0], c[1]])

    simps = [tuple(sorted(s)) for s in simps]

    tris = Delaunay(np.asarray(node_file[['X(px)', 'Y(px)']])).simplices
    tris = [tuple(sorted(t)) for t in tris]

    # Double check simplices actually should exist.
    print("\tChecking edges.")
    simps = [s for s in simps if s in tris]
    return np.asarray(simps)


def get_outer_edges(edges,
                    simplices):
    """
    This method sifts through the pairs of indices (edges) and checks how many times this edge occurs in a Delaunay
    triangle (simplice). If the edge is an outer edge it will only be contained in one triangle, otherwise it will be
    contained in two.
    Note that this method gets all edges regardless of if there is only one graph or multiple disconnected graphs.
    :param edges: This is an array of the index pairs. (shape = (n_pairs, 2))
    :type edges: np.ndarray
    :param simplices: This is an array containing the node indices that are contained in the Delaunay triangles.
        (shape = (n_triangles, 3))
    :type simplices: np.ndarray
    :return outer_pairs: A list of edges that lie on the outside of the graph.
    """

    # Convert the simplices into edges and then into tuples.
    s_edges0 = simplices[:, [0, 1]]
    s_edges1 = simplices[:, [0, 2]]
    s_edges2 = simplices[:, [1, 2]]

    s_edges = np.concatenate((s_edges0, s_edges1), axis=0)
    s_edges = np.concatenate((s_edges, s_edges2), axis=0)

    # Convert the edges to tuples.
    s_edges = [tuple(sorted(e)) for e in s_edges]
    edges = list(set([tuple(sorted(e)) for e in edges]))

    edge_counter = Counter(s_edges)
    min_counts = min(list(set(edge_counter.values())))
    outer_pairs = [p for p in edges if edge_counter[p] == min_counts]
    outer_pairs = np.asarray(outer_pairs)
    return outer_pairs


def outer_edges_to_path(nodes,
                        edges):
    """
    This method takes all the outside edges and puts the indices or each node lying on those edges in order.
    :param nodes: An array containing the coordinates of the outer graph edges.
    :type nodes: np.ndarray
    :param edges: A list of the outer graph edges.
    :return outlines: A list of lists. Each list contains the nodes belonging to outline.
    """

    outlines = []
    # Iterate through all the possible nodes. If the node doesn't yet exist in an outline then get the polygon it
    # belongs to.
    for node in np.unique(edges):
        # sum(polygons, []) just combines all list in a list of lists.
        if node not in set(sum(outlines, [])):
            outline = [node]
            while outline[-1] != outline[0] or len(outline) == 1:
                # Get all the edges containing the last node in the outline.
                pairs = [pair for pair in edges if outline[-1] in pair]
                # Order them so it is in the order [last node, next node]
                pairs = [p.tolist() if p[0] == outline[-1] else [p[1], p[0]] for p in pairs]
                # If it is the first node then we will calculate the angle from vertical up/north.
                if len(outline) == 1:
                    # Make the last edge just a vertical line.
                    # The fake node is 1 unit in the y-axis lower so moving to the current node is positive.
                    north = nodes[outline[-1], :] + np.array([0, 1])
                    last_edge = -(np.array([north, nodes[outline[-1], :]]) - nodes[outline[-1], :])
                else:
                    # Remove the previous pair from the list of possible next nodes.
                    pairs = [p for p in pairs if outline[-2] not in p]
                    last_edge = -(nodes[[outline[-2], outline[-1]], :] - nodes[outline[-1], :])
                # Position the start of each pair of nodes in each edge at the origin which is the position of the
                # last node.
                o_nodes = nodes - nodes[outline[-1]]
                angles = []
                for pair in pairs:
                    dot = (last_edge[0][0]*o_nodes[pair, :][1][0]) + (last_edge[0][1]*o_nodes[pair, :][1][1])
                    # dot = np.dot(last_edge[1], o_nodes[pair[1]])
                    # det = np.linalg.det(last_edge - o_nodes[pair, :])
                    det = (last_edge[0][0] * o_nodes[pair[1], :][1]) - (last_edge[0][1] * o_nodes[pair[1], :][0])
                    # We add pi because the angle would've been in the
                    # range [-pi, pi] but we want it in the
                    # range [0, pi].
                    theta = np.arctan2(det, dot) + np.pi
                    angles.append(theta)
                min_pair = pairs[angles.index(min(angles))]
                outline.append(min_pair[1])
            outlines.append(outline)
    return outlines


def graph_to_outline(node_file,
                     edge_file,
                     to_path=False,
                     threshold=None,
                     poly_threshold=None,
                     from_edges=False,
                     save=None
                     ):
    """
    This method is a combined method that takes the edge and node files, finds the outside edges
    which are those who only contribute to one triangle in Delaunay triangulation. It then orders
    these edges. Then it can save the points in this path to an .xml
    so that it can be displayed in MIM. It can be specified to convert this to a path or keep it in
    raw data form.
    *Note that the edges must've been formed using Delaunay triangulation.*
    :param node_file: The file containing the node information, specifically the coordinates of the nodes.
    :type node_file: pd.DataFrame or str
    :param edge_file: The file containing the edge indices and the distances for each edges.
    :type edge_file: pd.DataFrame or str
    :param to_path: If this is true then return the outlines as shapely polygons, otherwise return the raw coordinates.
            (default :obj: False)
    :type to_path: bool
    :param threshold: This is a distance threshold. Edges with a length greater than this will be removed. In um.
            (default :obj: `None`)
    :type threshold: float or int
    :param poly_threshold: Sometimes there are edges that are flagged as outlines because they were detected as not
        being within the list of Delaunay triangles, even though they should be. So we can remove outlines composed
        of a number of edges less than specified here.
    :type poly_threshold: int
    :param from_edges: If specified as True this will reconstruct the Delaunay simplices from the edge connections.
        If False then the simplices will be determined by applying the scipy Delaunay package to determine the
        simplices.
    :type from_edges: bool
    :param save: If save is not None then specify the name/path of the output file. (default :obj: `None`)
    :type save: str
    :return paths:
    """

    node_file = pd.read_csv(node_file) if type(node_file) == str else node_file
    edge_file = pd.read_csv(edge_file) if type(edge_file) == str else edge_file
    step_count = 0

    # Firstly convert the current data back into Delaunay triangle format.
    print(f'{step_count}. Reconstructing Delaunay triangles.')
    if from_edges:
        triangles = get_delaunay_from_edges(node_file,
                                            edge_file,
                                            threshold=threshold)
    else:
        triangles = delaunay_with_threshold(node_file,
                                            threshold=threshold)
    step_count += 1

    print(f'{step_count}. Getting outer edges.')
    outer_edges = get_outer_edges(np.asarray(edge_file[['source', 'target']]),
                                  triangles)
    step_count += 1

    print(f'{step_count}. Ordering edge coordinates.')
    node_coords = np.asarray(node_file[['X(px)', 'Y(px)']])
    path_edges = outer_edges_to_path(node_coords,
                                     outer_edges)

    if poly_threshold is not None:
        path_edges = [path for path in path_edges if len(path) > poly_threshold]

    paths = [node_coords[path, :] for path in path_edges]
    step_count += 1

    if save is not None:
        print(f"{step_count}. Writing to xml.")
        step_count += 1
        name = get_id(save)
        root = ET.Element("xml")
        root2 = ET.Element('Annotations', LayerName='Graph_Outlines')
        root.append(root2)
        for i, path in enumerate(paths):
            centre = path.mean(axis=0)
            curve_el = ET.SubElement(root2,
                                     f"Curve",
                                     name=f"{name}_{i}",
                                     description="",
                                     colour="#00ff00",
                                     cy=f"{centre[1]:.2f}",
                                     cx=f"{centre[0]:.2f}",
                                     zoom="1"
                                     )
            for pair in path:
                vertex_el = ET.SubElement(curve_el,
                                          "Vertex",
                                          y=str(pair[1]),
                                          x=str(pair[0]),
                                          )
        tree = ET.ElementTree(root)
        save = save + ".xml" if "xml" not in save else save
        with open(save, "wb") as files:
            tree.write(files)

    if to_path:
        print(f'{step_count}. Converting edges to shapely polygons.')
        paths = [get_polygon(path) for path in paths]
        step_count += 1

    return paths


def get_node_connections(idx,
                         node_df,
                         edge_df,
                         ):
    """ This method takes the index of a supernode and then uses the
    edge and node files to find all connections with nodes in the
    previous supernode level and return a dataframe containing those connects.
    :param idx: The index of the node you want to find connections for.
    :type idx: int
    :param node_df: The dataframe containing nodes and features.
    :type node_df: pd.dataframe
    :param edge_df: The dataframe containing edges and features.
    :type edge_df: pd.dataframe
    :return:
    """

    supernode = node_df.loc[idx]['SN']
    source_sn = edge_df[edge_df['source'] == idx]
    ids = source_sn['target'].tolist()
    keep_ids = []
    for i in ids:
        if node_df['SN'].iloc[i] == supernode-1:
            keep_ids.append(i)

    return node_df.loc[keep_ids]


def remove_outer_supernodes(node_path,
                            edge_path):
    """
    This method takes the cell graph with one level of supernodes. It then removes the supernodes that are not connected
    to the cell graph at all. The new node and edge files without these supernodes are returned.
    :param node_path: The path to the file containing the coordinates of the cell graph and supernodes.
    :type node_path: str
    :param edge_path: The path to the file containing the connections for the cell graph and supernodes.
    :type edge_path: str
    :return:
    """

    def replace_id(x, old_list, new_list):
        if x in old_list:
            index = old_list.index(x)
            return new_list[index]
        elif x not in old_list and x < min(new_list):
            return x
        else:
            return -1

    if type(node_path) == str:
        node_file = pd.read_csv(node_path)
        edge_file = pd.read_csv(edge_path)
    else:
        node_file = node_path
        edge_file = edge_path

    # Get a list of the supernodes that have connections to the cell graph.
    sn0s = node_file[node_file['SN'] == 0]
    sn1s = node_file[node_file['SN'] == 1]

    # Iterate through each supernodes and if it only contains connections to other supernodes, then do not add it to
    # the keep list.
    keep_ids = []
    for idx in sn1s.index.tolist():
        temp_df = edge_file[edge_file['source'] == idx]
        if temp_df['target'].min() <= len(sn0s):
            keep_ids.append(idx)

    reduced_sn1s = sn1s.loc[keep_ids]
    old_ids = reduced_sn1s.index.tolist()
    new_ids = [i for i in range(len(sn0s.index.tolist()), len(sn0s.index.tolist()) + len(reduced_sn1s))]
    reduced_sn1s.index = new_ids
    node_file = pd.concat([sn0s, reduced_sn1s], axis=0)

    sn0_edges = edge_file[edge_file['SN'] == 0]
    sn1_edges = edge_file[edge_file['SN'] == 1]
    sn1_edges['source'] = sn1_edges['source'].apply(lambda x: replace_id(x, old_ids, new_ids))
    sn1_edges['target'] = sn1_edges['target'].apply(lambda x: replace_id(x, old_ids, new_ids))
    sn1_edges = sn1_edges[sn1_edges['source'] != -1]
    sn1_edges = sn1_edges[sn1_edges['target'] != -1]
    edge_file = pd.concat([sn0_edges, sn1_edges])

    return node_file, edge_file



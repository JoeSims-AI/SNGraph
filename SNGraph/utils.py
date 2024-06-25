import time
from os.path import *
import pandas as pd
from glob import glob


def get_formatted_date() -> str:
    """ When called this method will return the date in
    the format of ddmmyyyy

    :return ddmmyyyy:
    """
    time_obj = time.gmtime()
    # Day and month need to be formatted to be dd/mm even if day and month are <10.
    day = str(time_obj.tm_mday)
    day = f"0{day}" if len(day) == 1 else day
    month = str(time_obj.tm_mon)
    month = f"0{month}" if len(month) == 1 else month
    return f"{day}{month}{time_obj.tm_year}"


def get_id(path) -> str:
    """
    This takes the path and splits it into the directories and file name.
    It then returns just the file name with out the file extension.

    :param path: The path to the file.
    :type path: str
    :return file_name:
    """
    return splitext(split(path)[1])[0]


def order_files(node_dir,
                edge_dir):
    """
    This method takes a list of the node csv files and edge csv files in their directories and
    then puts them in their corresponding order within the lists.

    :param node_dir: The directory for the node files.
    :type node_dir: list
    :param edge_dir: The directory for the edge files.
    :type edge_dir: list
    :return node_files: These are the names of the files containing node information.
    :return edge_files: There are the names of the files containing edge information in
        the same order as the node files.
    """
    if not isdir(node_dir):
        raise Exception(f'Directory {node_dir} does not exist.')
    if not isdir(edge_dir):
        raise Exception(f'Directory {node_dir} does not exist.')

    node_files = [f for f in glob(node_dir + '/*')]
    edge_files = [f for f in glob(edge_dir + '/*')]

    node_ids = [get_id(f) for f in node_files]
    edge_ids = [get_id(f) for f in edge_files]

    ordered_edge_files = []
    for n, n_id in zip(node_files, node_ids):
        for e, e_id in zip(edge_files, edge_ids):
            if n_id == e_id:
                ordered_edge_files.append(e)

    if len(node_files) == 0 or len(ordered_edge_files) == 0:
        raise Exception(
            f"There is a problem with the number of files. n_node_files = {len(node_files)}",
            f"n_edge_files = {(len(ordered_edge_files))}")

    return node_files, ordered_edge_files


def round_to(input_value,
             round_value) -> float:
    """ This method takes a value outside of the range [-1,1] and rounds
    it to the closest value.

    Examples:
        round(5374, 10) -> 5370
        round(5374, 500) -> 5500
        round(5374, 1000) -> 5000

    :param input_value: Value to be rounded.
    :type input_value: int
    :param round_value: Factor of nearest number to be rounded to.
    :type round_value: int
    :return rounded_value:
    """
    return round(input_value/round_value) * round_value


def floor_to(input_value,
             round_value) -> float:
    """ In some cases you don't want to round to the nearest factor.
    Instead you might specifically want to round up or down to create
    a positive buffer around the values. This method is for rounding
    down to the nearest factor.

    Examples:
        floor_to(5374, 100) = 57300
        floor_to(5374, 5) = 5370

    :param input_value: Value to be rounded.
    :type input_value: int
    :param round_value: Factor of nearest number to be rounded to.
    :type round_value: int
    :return rounded_value:
    """
    rounded = round_to(input_value, round_value)
    return rounded if rounded <= input_value else rounded - round_value


def ceil_to(input_value,
            round_value) -> float:
    """ In some cases you don't want to round to the nearest factor.
    Instead you might specifically want to round up or down to create
    a positive buffer around the values. This method is for rounding
    up to the nearest factor.

    Examples:
        floor_to(5374, 100) = 57400
        floor_to(5374, 5) = 5375

    :param input_value: Value to be rounded.
    :type input_value: int
    :param round_value: Factor of nearest number to be rounded to.
    :type round_value: int
    :return rounded_value:
    """
    rounded = round_to(input_value, round_value)
    return rounded if rounded >= input_value else rounded + round_value


def calculate_sf(file,
                 numerator="px",
                 denominator="um"):
    """
    This function finds the linear scale factor between units.
    For example, when dealing with node locations within a digital pathology image, the location is specified in
    pixels (px) and micrometers (um). When dealing with images and physical sizes, we want to convert between them
    so we use a scale factor to do this.

    :param file: Either the path or the pandas df for the node locations.
    :type file: str or pandas.DataFrame
    :param numerator: The target units. (default :obj:`px`).
    :type numerator: str
    :param denominator: The input units. (default :obj:`um`).
    :type denominator: str
    :return scale_factor:
    """

    if type(file) == str:
        if not isfile(file):
            raise Exception(f'File does not exist: {file}')
        else:
            file = pd.read_csv(file)

    xsf = file[f'X({numerator})'].mean() / file[f'X({denominator})'].mean()
    ysf = file[f'Y({numerator})'].mean() / file[f'Y({denominator})'].mean()
    return xsf, ysf

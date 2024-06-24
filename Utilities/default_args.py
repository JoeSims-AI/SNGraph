"""
    This takes in the parameter file presented and sets missing parameters to the default.
    It also sets up the projects locations.
"""

from os.path import join


def default_true(dic, param):
    return False if param in dic and dic[param].lower() == "false" else True


def default_false(dic, param):
    return True if param in dic and dic[param].lower() == "true" else False


def default_float(dic, param, val):
    return float(dic[param]) if param in dic else val


def default_int(dic, param, val):
    return int(dic[param]) if param in dic else val


def default_str(dic, param, arg):
    return dic[param] if param in dic else arg


def get_args(path):
    """
    This method retrieves the arguments from a txt file to use.

    :param path: The path to the txt file.
    :type path: str
    :return params: A dictionary containing the parameters.
    """

    f = open(path, 'r')
    lines = f.readlines()

    # Remove new line token
    lines = [line.strip('\n') for line in lines]

    # Collect parameters in a dictionary.
    params = {}
    for line in lines:
        line_split = line.split(" ")
        # If there is a list specified, then we need to convert it from a long string to a list.
        params[line_split[0]] = line_split[1] if len(line_split) == 2 else [l.strip(',') for l in line_split[1:]]

    return params


def get_params(path):

    params = get_args(path)

    project_dir = join(params["PROJECT_DIR"], params["NAME"])

    param_dict = {"NAME": params["NAME"],
                  "PROJECT_DIR": project_dir,
                  "GRAPH_DIR": join(project_dir, "Graphs"),
                  "SN0_DIR": join(project_dir, "Graphs", "SN0"),
                  "SN1_DIR": join(project_dir, "Graphs", "SN1"),
                  "SN2_DIR": join(project_dir, "Graphs", "SN2"),
                  "LOG_DIR": join(project_dir, "LogFiles"),
                  "LOSS_DIR": join(project_dir, "LossFiles"),
                  "MODEL_DIR": join(project_dir, "Models"),
                  "EVAL_DIR": join(project_dir, "EvaluationFiles"),
                  "CV_PATH": params["CV_PATH"],
                  "OUTLINE_DIR": params["OUTLINE_DIR"],
                  "EPOCHS": default_int(params, "EPOCHS", 1000),
                  "SAVE_EPOCHS": default_int(params, "SAVE_EPOCHS", 100),
                  "NORMALISE_CM": default_true(params, "NORMALISE_CM"),
                  "LAYOUT": default_str(params, "LAYOUT", "regular"),
                  "SEPARATION": default_int(params, "SEPARATION", 140),
                  "RADIUS": default_int(params, "RADIUS", 1),
                  "XPX_COL": default_str(params, "XPX_COL", "X(px)"),
                  "YPX_COL": default_str(params, "YPX_COL", "Y(px)"),
                  "X_COL": default_str(params, "XPX_COL", "X(um)"),
                  "Y_COL": default_str(params, "YPX_COL", "Y(um)"),
                  "FOLD": default_int(params, "FOLD", 0),
                  "IN_FEATURES": params["IN_FEATURES"],
                  "OUT_FEATURES": params["OUT_FEATURES"],
                  "EDGE_FEATURES": params["EDGE_FEATURES"],
                  "N_MESSAGE_PASSING": default_int(params, "N_MESSAGE_PASSING", 1),
                  "VALIDATE": default_true(params, "VALIDATE"),
                  "NOSIE_CELLS": default_float(params, "NOISE_CELLS", 0.1),
                  "NOISE_ATT": default_float(params, "NOISE_ATT", 1),
                  "NOISE": default_float(params, "NOISE", 1),
                  "THRESHOLD": default_float(params, "NOISE_CELLS", None)
                  }
    return param_dict
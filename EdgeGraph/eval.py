import torch
from torch import Tensor
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sksurv.metrics import concordance_index_censored


def calculate_cindex(model,
                     dataloader,
                     device='cpu'):
    """
    This method calculates the concordance index (c-index).
    This method is applicable to censored data.
    This method uses the relative hazard which is valid under the assumption that the output follows a
    proportional hazards model.

    :param model: The deep learning survival model.
    :type model: torch_geometric.model
    :param dataloader: The data loader containing the graphs to be evaluated.
    :type dataloader: torch_geometric.loader.DataLoader
    :param device: CUDA or CPU. (default :obj:`cpu`)
    :type device: str
    :return:
    """

    model = model.to(device)

    # Run the model on all data to get outputs.
    durations = []
    events = []
    y_preds = []
    for batch in dataloader:
        events.extend(batch.y[:, 0].tolist())
        durations.extend(batch.y[:, 1].tolist())
        batch = batch.to(device)
        with torch.no_grad():
            y_pred, _ = model(batch)

        y_pred = y_pred.detach().squeeze().tolist()
        if type(y_pred) == float:
            y_pred = [y_pred]
        y_preds.extend(y_pred)

    grouping = pd.DataFrame({'status': events,
                             'time': durations,
                             'hazard': y_preds})

    c_index, _, _, _, _ = concordance_index_censored(grouping['status'] == 1,  # needs to be boolean
                                                     grouping['time'],
                                                     grouping['hazard'])
    return c_index



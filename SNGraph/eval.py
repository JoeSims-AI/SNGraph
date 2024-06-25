import torch
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored
from torch import Tensor


def precision(cm) -> float:
    """
    Calculate the precision from the confusion matrix.

    :param cm: confusion matrix.
    :type cm: np.ndarray
    :return precision:
    """
    return cm[0, 0] / (cm[0, 0] + cm[1, 0])


def recall(cm) -> float:
    """
    Calculate the recall from the confusion matrix.
    :param cm: confusion matrix.
    :type cm: np.ndarray
    :return recall:
    """
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])


def f1(cm) -> float:
    """
    Calculate the f1 score from the confusion matrix.

    :param cm: confusion matrix.
    :type cm: np.ndarray
    :return f1:
    """
    p = precision(cm)
    r = recall(cm)
    return 2 * ((p * r) / (p + r))


def balanced_acc(cm) -> float:
    """
    This method calculates the recalls for each class and takes the average
    and apparently this is the balanced accuracy.

    :param cm: confusion matrix.
    :type cm: np.ndarray
    :return balanced_acc:
    """
    recalls = []
    for i in range(cm.shape[0]):
        if cm[i, :].sum() > 0:
            recalls.append(cm[i, i] / cm[i, :].sum())
        else:
            recalls.append(0)
    return sum(recalls) / len(recalls)


def format_cm(cm,
              dp=None):
    """
    This takes confusion matrices and prints them in a nice manner.
    If the dp is specified then the values will be rounded to this many
    significant figures.

    :param cm: The confusion matrix.
    :type cm: np.ndarray
    :param dp: The number of places to round to.
        (default :obj: `3`)
    :type dp: int
    """
    if cm.max() < 1:
        dp = 3 if dp is None else dp
        cm = np.round(cm, dp)

    cm_string = ''
    max_len = len(str(cm.max()))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            elem = str(cm[i, j])
            space = ' ' * (max_len - len(elem) + 2)
            cm_string += space + elem
        cm_string += '\n'

    return cm_string


def mean_acc(vals):
    """
        This returns the mean and standard error from values in a list.
    :param vals:
    :return:
    """
    vals = np.asarray(vals)
    mean_a = vals.mean()
    a_unc = np.std(vals) / np.sqrt(len(vals))
    return mean_a, a_unc


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


def cox_loss_ph(h, events):
    """
        This function was acquired from pycox. https://github.com/havakv/pycox
        Requires the input to be sorted by descending duration time.
        See DatasetDurationSorted.
        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.
        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """
    riskmax = torch.nn.functional.normalize(h, p=2, dim=0)
    log_risk = torch.log((torch.cumsum(torch.exp(riskmax), dim=0)))
    uncensored_l = torch.add(riskmax, -log_risk)
    resize_censors = events.resize_(uncensored_l.size()[0], 1)
    censored_likelihood = torch.mul(uncensored_l, resize_censors)
    loss = -torch.sum(censored_likelihood) / float(events.nonzero().size(0))
    return loss


def cox_loss_sorted(h, events, durations):

    idx = durations.sort(descending=True, dim=0)[1]
    events = events[idx]
    h = h[idx]
    return cox_loss_ph(h, events)

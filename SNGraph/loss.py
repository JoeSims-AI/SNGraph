import torch
from torch import Tensor


def weighted_mean_squared_error(target,
                                output,
                                weights,
                                device='cpu',
                                ):
    """
    Weighted, masked mean squared error.
    The masking allows the loss to be assessed on specified nodes only.
    It ignores unlabelled nodes in the training process.
    Because unlabelled nodes will be a large portion of the graph >90% total nodes,
    the model wouldn't be able to learn the labelled classes as they will be too small.

    :param target:  A tensor containing the ground trust nodes labels
    :type target: tensor
    :param output: A tensor containing the output node labels from the model.
    :type output: tensor
    :param weights: A list containing the loss weights for the output classes.
    :type weights: list
    :param device: Is the device cpu or cuda?
    :return wmse: A 1D tensor of the weighted masked mean squared error.
    """
    weights = torch.tensor([weights], dtype=torch.float).to(device)
    weights = weights * weights
    weights_2d = torch.tile(weights, (target.size()[0], 1))

    mask = torch.sum(target, 1)
    mask_2d = torch.tile(mask, (len(weights), 1))

    mw = torch.transpose(mask_2d, 0, 1) * weights_2d

    wmse = torch.mean(mw * (output - target) ** 2)
    return wmse


def cox_loss_ph(h,
                events) -> Tensor:
    """
    With the data being sorted in descending duration, we can calculate the cumulative sum which can
    act as the risk set even though it is not the true risk set.

    :param h: A tensor containing the output from the model containing risks for multiple instances.
    :type h: Tensor
    :param events: A tensor composed of 0s and 1s representing censored and uncensored events, respectively.
    :type events: Tensor
    :return loss:
    """

    risk_max = torch.nn.functional.normalize(h, p=2, dim=0)
    log_risk = torch.log((torch.cumsum(torch.exp(risk_max), dim=0)))
    uncensored_l = torch.add(risk_max, -log_risk)
    resize_censors = events.resize_(uncensored_l.size()[0], 1)
    censored_likelihood = torch.mul(uncensored_l, resize_censors)
    loss = -torch.sum(censored_likelihood) / float(events.nonzero().size(0))
    return loss


def cox_loss_sorted(h,
                    events,
                    durations) -> Tensor:
    """
    This method orders the output from the model in descending duration.

    :param h: A tensor containing the output from the model containing risks for multiple instances.
    :type h: Tensor
    :param events: A tensor composed of 0s and 1s representing censored and uncensored events, respectively.
    :type events: Tensor
    :param durations: A tensor composed of the times to events. These should be in the corresponding order to the
        events.
    :return loss:
    """

    idx = durations.sort(descending=True, dim=0)[1]
    events = events[idx]
    h = h[idx]
    return cox_loss_ph(h, events)
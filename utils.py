"""Loss and metric utilities used by the training script."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from lifelines.utils import concordance_index


class Regularization(object):
    def __init__(self, order, weight_decay):
        super().__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


def logsumexp(input_tensor: Tensor, mask: Tensor = None, dim: int = None,
              keepdim: bool = False) -> Tensor:
    """Compute a numerically stable, optionally masked log-sum-exp."""
    if dim is None:
        input_tensor, dim = input_tensor.view(-1), 0
    max_value, _ = torch.max(input_tensor, dim=dim, keepdim=True)
    input_tensor = input_tensor - max_value
    res = torch.exp(input_tensor)
    if mask is not None:
        res = res * mask
    res = torch.log(torch.sum(res, dim=dim, keepdim=keepdim) + 1e-8)
    return res + max_value.squeeze(dim) if not keepdim else res + max_value


class NegativeLogLikelihood(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L2_reg = config['l2_reg']
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, theta, time, delta, model, device):
        time = time.view(-1, 1)
        delta = delta.view(-1, 1)
        time = time.reshape(-1)
        theta = theta.reshape(-1, 1)
        risk_mat = (time >= time[:, None]).float()
        loss_cox = (theta.reshape(-1)
                    - logsumexp(theta.T, mask=risk_mat, dim=1)) * delta.reshape(-1)
        loss_cox = loss_cox.sum() / (delta.sum() + 1e-10)
        l2_loss = self.reg(model)
        return -loss_cox + l2_loss


def c_index(risk_pred, y, e):
    """Concordance index; accepts tensors or numpy arrays."""
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

from typing import Dict, Union, Sequence

import torch
import torch.nn as nn

import kornia.losses as krloss

from mmdepth.registry import LOSSES

# see: https://pytorch.org/docs/stable/nn.html#loss-functions
for module in [
    nn.L1Loss, nn.MSELoss, nn.CrossEntropyLoss, nn.CTCLoss,
    nn.NLLLoss, nn.PoissonNLLLoss, nn.KLDivLoss, nn.BCELoss,
    nn.BCEWithLogitsLoss, nn.MarginRankingLoss, nn.HingeEmbeddingLoss,
    nn.MultiLabelMarginLoss, nn.SmoothL1Loss, nn.SoftMarginLoss,
    nn.MultiLabelSoftMarginLoss, nn.CosineEmbeddingLoss, nn.MultiMarginLoss,
    nn.TripletMarginLoss, nn.TripletMarginWithDistanceLoss
]:
    LOSSES.register_module(module=module)

# see: https://kornia.readthedocs.io/en/stable/losses.html
for module in [
    krloss.SSIMLoss, krloss.SSIM3DLoss, krloss.MS_SSIMLoss,
    krloss.FocalLoss, krloss.CharbonnierLoss, krloss.WelschLoss,
    krloss.CauchyLoss, krloss.GemanMcclureLoss
]:
    LOSSES.register_module(module=module)


# distribution loss
@LOSSES.register_module()
class JSDivLoss(nn.Module):
    """ Jensen-Shannon divergence loss
    see: https://kornia.readthedocs.io/en/stable/losses.html
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        return krloss.js_div_loss_2d(pred, target, self.reduction)


@LOSSES.register_module()
class KLDivLoss(nn.Module):
    """Kullback-Leibler divergence loss
    see: https://kornia.readthedocs.io/en/stable/losses.html
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        return krloss.kl_div_loss_2d(pred, target, self.reduction)


#


@LOSSES.register_module()
class LogCoshLoss(nn.Module):
    """Log-cosh loss function.

    Computes the logarithm of the hyperbolic cosine of the prediction error.
    """

    def __init__(self, reduction):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(pred - target))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction is None:
            return loss
        else:
            raise NotImplementedError(f"reduction:{self.reduction} "
                                      f"is not implemented")


def build_loss(cfg: Dict) -> nn.Module:
    """Build loss function.

    Args:
        cfg (dict): The loss function config, which should contain:
            - type (str): Loss function type.
            - loss_args: Args needed to instantiate a loss function.

    Returns:
        nn.Module: Created loss function.
    """
    return LOSSES.build(cfg)

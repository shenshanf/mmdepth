from typing import Dict, Optional

import torch
from mmengine.model import BaseModule
from torch import nn as nn

from mmdepth.structures import DispMap
from mmdepth.models.modules.bricks import build_loss
from mmdepth.registry import LOSSES, MODELS


@LOSSES.register_module()
class DispRegressionLoss(BaseModule):
    """Disparity Regression Loss Module.

    This module computes the regression loss between predicted and ground truth disparity maps,
    supporting various loss functions like L1Loss, MSELoss, etc.
    """

    def __init__(self, criterion_cfg: Dict, mapper_cfg: Optional[Dict] = None):
        super().__init__()
        # e.g. L1Loss, MSELoss, SmoothL1Loss, HuberLoss, LogCoshLoss
        self.criterion: nn.Module = build_loss(criterion_cfg)
        if mapper_cfg is None:
            self.mapper = nn.Identity()
        else:
            self.mapper = MODELS.build(mapper_cfg)  # e.g. log function

    def forward(self, pr_disp: DispMap, gt_disp: DispMap, conf_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function for computing loss.

        Args:
            pr_disp (DispMap): Predicted disparity map.
                data:
                v_mask: mask of valid region
            gt_disp (DispMap): Ground truth disparity map.
                data:
                v_mask: mask of valid region
            conf_map (torch.Tensor): confidence map
        Returns:
            torch.Tensor: Computed regression loss.
        """
        # get valid mask
        pr_v_mask = pr_disp.v_mask
        gt_v_mask = gt_disp.v_mask

        # valid mask
        if pr_v_mask is not None and gt_v_mask is None:
            v_mask = pr_v_mask
        elif pr_v_mask is None and gt_v_mask is None:
            v_mask = torch.ones_like(pr_disp.data, dtype=torch.bool)
        elif pr_v_mask is None and gt_v_mask is not None:
            v_mask = gt_v_mask
        else:
            v_mask = pr_v_mask & gt_v_mask

        # compute loss
        if conf_map is not None:
            regression_loss = self.criterion(
                self.mapper(pr_disp.data[v_mask]) * conf_map[v_mask],
                self.mapper(gt_disp.data[v_mask]) * conf_map[v_mask]
            )
        else:
            regression_loss = self.criterion(
                self.mapper(pr_disp.data[v_mask]),
                self.mapper(gt_disp.data[v_mask])
            )
        return regression_loss

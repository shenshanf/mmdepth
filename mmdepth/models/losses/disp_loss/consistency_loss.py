from typing import Optional, Dict

from mmengine.model import BaseModule
import torch
from torch import nn as nn

from mmdepth.models.modules.bricks import build_loss
from mmdepth.models.modules.opts import WarpOpt2D
from mmdepth.registry import LOSSES

from mmdepth.structures import DispMap


@LOSSES.register_module()
class DispReconstructLoss(BaseModule):
    _default_warp_cfg = dict(mode='bilinear', padding_mode='border', use_mask=False)
    _default_criterion_cfg = dict(type='SSIMLoss', data_range=1.0, kernel_size=(11, 11),
                                  sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True)
    def __init__(self, warp_cfg: Optional[Dict], criterion_cfg: Optional[Dict]):
        super().__init__()
        if warp_cfg is None:
            warp_cfg = self._default_warp_cfg
        if criterion_cfg is None:
            criterion_cfg = self._default_criterion_cfg

        self.warp_opt = WarpOpt2D(**warp_cfg)
        self.criterion: nn.Module = build_loss(criterion_cfg)

    def forward(self, pr_disp: DispMap, feat_left, feat_right):
        """

        Args:
            pr_disp: [b,1,h,w]
            feat_left: [b,c,h,w]: image/learned feature/gradient map/
            feat_right: [b,c,h,w]

        Returns:

        """
        assert not pr_disp.is_sparse
        feat_left_prime = self.warp_opt(feat_right, -pr_disp.data)
        reconstruct_loss = self.criterion(feat_left, feat_left_prime)
        return reconstruct_loss


@LOSSES.register_module()
class DispLoopConsLoss(BaseModule):
    """Loop consistency loss for disparity estimation.

    计算方式:
    1. 左图 -> 右图 -> 左图的重投影
    2. 比较原始左图与重投影后的左图
    """
    _default_warp_cfg = dict(mode='bilinear', padding_mode='border', use_mask=False)
    _default_criterion_cfg = dict(type='L1Loss', reduction='mean')

    def __init__(self, warp_cfg: Optional[Dict] = None,
                 criterion_cfg: Optional[Dict] = None):
        super().__init__()
        if warp_cfg is None:
            warp_cfg = self._default_warp_cfg
        if criterion_cfg is None:
            criterion_cfg = self._default_criterion_cfg

        self.warp_opt = WarpOpt2D(**warp_cfg)
        self.criterion: nn.Module = build_loss(criterion_cfg)

    def forward(self, pr_disp_left: DispMap, pr_disp_right: DispMap,
                feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """Forward function for loop consistency loss.

        Args:
            pr_disp_left (DispMap): 左视图视差图
            pr_disp_right (DispMap): 右视图视差图
            feat_left (torch.Tensor): 左图特征, shape (B, C, H, W)
            feat_right (torch.Tensor): 右图特征, shape (B, C, H, W)

        Returns:
            torch.Tensor: 循环一致性损失值
        """
        assert not pr_disp_left.is_sparse
        assert not pr_disp_right.is_sparse

        # 左 -> 右
        feat_right_prime = self.warp_opt(feat_left, pr_disp_left.data)

        # 右 -> 左
        feat_left_loop = self.warp_opt(feat_right_prime, -pr_disp_right.data)

        # 计算循环一致性损失
        loop_cons_loss = self.criterion(feat_left, feat_left_loop)

        return loop_cons_loss


@LOSSES.register_module()
class DispLRCheckLoss(BaseModule):
    """Left-right consistency check loss for disparity estimation.

    计算左视差图和右视差图的一致性:
    L = |D_left(p) - D_right(p - D_left(p))|
    """
    _default_warp_cfg = dict(mode='bilinear', padding_mode='border', use_mask=False)
    _default_criterion_cfg = dict(type='SSIMLoss', data_range=1.0, kernel_size=(11, 11),
                                  sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True)

    def __init__(self, warp_cfg: Optional[Dict] = None,
                 criterion_cfg: Optional[Dict] = None):
        super().__init__()
        if warp_cfg is None:
            warp_cfg = self._default_warp_cfg
        if criterion_cfg is None:
            criterion_cfg = self._default_criterion_cfg

        self.warp_opt = WarpOpt2D(**warp_cfg)
        self.criterion: nn.Module = build_loss(criterion_cfg)


    def forward(self, pr_disp_left: DispMap, pr_disp_right: DispMap) -> torch.Tensor:
        """Forward function for left-right consistency check loss.

        Args:
            pr_disp_left (DispMap): 左视图视差图
            pr_disp_right (DispMap): 右视图视差图

        Returns:
            torch.Tensor: 左右一致性损失值
        """
        assert not pr_disp_left.is_sparse
        assert not pr_disp_right.is_sparse

        # 将右视差图投影到左视图坐标系
        pr_disp_right_to_left = self.warp_opt(pr_disp_right.data,
                                              -pr_disp_left.data)

        # 计算左右视差的差异
        lr_diff = self.criterion(pr_disp_left.data, pr_disp_right_to_left)

        # 应用reduction
        if self.reduction == 'none':
            return lr_diff
        elif self.reduction == 'mean':
            return lr_diff.mean()
        elif self.reduction == 'sum':
            return lr_diff.sum()
        else:
            raise ValueError(f'Invalid reduction mode: {self.reduction}')

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from kornia.filters import SpatialGradient

from mmdepth.structures import DispMap
from mmdepth.registry import LOSSES


@LOSSES.register_module()
class DispNeighborSmoothLoss(BaseModule):
    """Loss for enforcing smoothness between neighboring disparities.

    The loss computes: (1/N) * sum_p sum_{y in N_p} |d_p - d_y|
    where N_p is the neighborhood around pixel p.

    Args:
        kernel_size (int): Size of the neighborhood window. Must be odd. Default: 3
        reduction (str): Reduction method. Can be 'none', 'mean', 'sum'. Default: 'mean'
    """

    def __init__(
            self,
            kernel_size: int = 3,
            reduction: str = 'mean',
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f'kernel_size must be odd, got {kernel_size}')

        self.kernel_size = kernel_size
        self.reduction = reduction

    def forward(
            self,
            pr_disp: DispMap,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            pr_disp (DispMap):
                data: Predicted disparity map of shape (N, 1, H, W)
                v_mask: assert None
            mask (torch.Tensor, optional): Valid pixel mask of shape (N, 1, H, W)

        Returns:
            torch.Tensor: Computed loss
        """
        if not pr_disp.is_sparse:
            raise ValueError("Only sparse disparity maps are supported")

        pr_disp_data = pr_disp.data
        if pr_disp_data.dim() != 4:
            raise ValueError(f'pr_disp_data must have 4 dimensions, got {pr_disp_data.dim()}')

        # Get input shape
        B, C, H, W = pr_disp_data.shape

        if self.kernel_size > min(H, W):
            raise ValueError(f'kernel_size {self.kernel_size} is too large for input size {H}x{W}')

        # Get all neighbors for each pixel using unfold
        # Output shape: (b, c*k*k, h*w), c=1
        neighbors = F.unfold(
            pr_disp_data,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2
        )

        # Then reshape to (b, c=1, k*k, h*w)
        neighbors = neighbors.view(B, C, self.kernel_size ** 2, -1)

        # [b,1,h,w] -> [b,c=1,1,h*w]
        centers = pr_disp_data.view(B, C, 1, -1)

        # broadcast at k dimension
        # (b, c=1, k*k, h*w)
        diffs = torch.abs(centers - neighbors)

        # Sum over the neighborhood (k*k dimension)
        # (b,c=1,h,w)
        loss = diffs.sum(dim=2, keepdim=False).view(B, 1, H, W)

        if mask is not None:
            loss = loss * mask

        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            if mask is not None:
                return loss.sum() / (mask.sum() + 1e-8)
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f'reduction must be none/mean/sum, got {self.reduction}')


@LOSSES.register_module()
class DispGradientSmoothLoss(BaseModule):
    def __init__(self, gd_mode='diff', gd_order=2, reduction='mean'):
        super().__init__()
        self.gradient = SpatialGradient(mode=gd_mode, order=gd_order)
        self.gd_order = gd_order
        self.reduction = reduction

    def forward(self, pr_disp: DispMap):
        """

        Args:
            pr_disp:
                data: [b,1,h,w]

        Returns:

        """
        if not pr_disp.is_sparse:
            raise ValueError("Only sparse disparity maps are supported")

        pr_disp_data = pr_disp.data

        # [b,1,2,h,w] <- [b,1,h,w]
        smooth_loss = self.gradient(pr_disp_data).abs().pow(self.gd_order)
        # [b,1,h,w] sum all direction gradient
        smooth_loss = smooth_loss.sum(dim=2, keep_dim=False)
        if self.reduction == 'mean':
            return smooth_loss.mean()
        elif self.reduction == 'sum':
            return smooth_loss.sum()
        elif self.reduction is None:
            return smooth_loss
        else:
            raise NotImplementedError


@LOSSES.register_module()
class XAwareDispSmoothLoss(BaseModule):
    def __init__(self, beta=1.0, gd_mode='sobel', gd_order=1, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.gradient = SpatialGradient(mode=gd_mode, order=gd_order)
        self.gd_order = gd_order
        self.reduction = reduction

    def forward(self, pr_disp: DispMap, x: torch.Tensor):
        """

        Args:
            pr_disp:
                data: [b,1,h,w]
            x: [b,c,h,w], could be: image map, edge map ...

        Returns:

        """
        if not pr_disp.is_sparse:
            raise ValueError("Only sparse disparity maps are supported")

        pr_disp_data = pr_disp.data

        if pr_disp_data.shape[-2:] != x.shape[-2:]:
            raise ValueError(f"Shape mismatch: disp {pr_disp_data.shape} vs x {x.shape}")

        # [b,c=1,2,h,w] <- [b,c=1,h,w]
        gd_pr_disp = self.gradient(pr_disp_data).abs().pow(self.gd_order)
        # [b,c=1,2,h,w]<-[b,c,2,h,w] <- [b,c,h,w]
        gd_x = (self.gradient(x).abs().pow(self.gd_order)).mean(dim=1, keepdim=False)

        # [b,1,h,w]
        smooth_loss = (gd_pr_disp * torch.exp(-self.beta * gd_x)).sum(dim=2, keepdim=False)

        if self.reduction == 'mean':
            return smooth_loss.mean()
        elif self.reduction == 'sum':
            return smooth_loss.sum()
        elif self.reduction is None:
            return smooth_loss
        else:
            raise NotImplementedError

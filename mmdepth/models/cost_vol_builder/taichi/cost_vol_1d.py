import torch
import taichi as ti

from ..base import BaseCtVolBuilder
from .kernels import TAICHI_ARCH
from .functions import CorrCostVolFn


class CorrCtVolBuilder(BaseCtVolBuilder):
    @property
    def out_channels(self):
        return

    def __init__(self, disp_max=None, disp_min=0, step=1, act_cfg=None):
        if act_cfg is not None:
            raise NotImplementedError(f"inplace activate function is not supported currently")
        super().__init__(disp_max, disp_min, step, act_cfg)

    def forward(self, feat_left, feat_right) -> torch.Tensor:
        if feat_left.device != feat_right.device:
            raise ValueError(
                f"Device mismatch: left feature on {feat_left.device} but "
                f"right feature on {feat_right.device}"
            )
        if feat_left.device.type == 'cuda':
            assert TAICHI_ARCH == ti.cuda
        elif feat_left.device.type == 'cpu':
            assert TAICHI_ARCH == ti.cpu
        else:
            raise NotImplementedError(f"torch device:{feat_left.device} "
                                      f"is not consistent with taichi arch: {TAICHI_ARCH}")

        # Check if inputs are contiguous
        if not feat_left.is_contiguous():
            raise RuntimeError("Left feature must be contiguous")
        if not feat_right.is_contiguous():
            raise RuntimeError("Right feature must be contiguous")

        return CorrCostVolFn.apply(feat_left, feat_right,
                                   self.disp_max, self.disp_min, self.step, None)

    def __repr__(self):
        return 'taichi backend: corr_cost_volume'

from typing import Optional, Dict
from multimethod import multimethod
import torch

from mmdepth.models import CostVolume

from ..base import BaseDymCtVolBuilder, BaseCtVolLookup
from .match_sampler import UniformMatchSampler, DymUniformMatchSampler
from mmdepth.models.modules.opts import WarpOpt3D, LookUpOpt
from .registry import MATCH_FEAT


# class _BaseDymCtVolBuilder(BaseDymCtVolBuilder):
#     _default_match_feat_cfg = dict(
#         type='CorrMatchFeat',
#         reduce=True,
#         keep_dim=False
#     )
#
#     def __init__(self, match_feat_cfg: Optional[Dict] = None, sample_mode='bilinear',
#                  padding_mode='border'):
#         super().__init__()
#         self.warp = WarpOpt3D(mode=sample_mode,
#                               padding_mode=padding_mode,
#                               use_mask=False)
#         if match_feat_cfg is None:
#             match_feat_cfg = self._default_match_feat_cfg
#         self.match_feat = MATCH_FEAT.build(match_feat_cfg)
#
#     def _forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor,
#                  sample_grid: torch.Tensor) -> CostVolume:
#         """
#
#         Args:
#             feat_left:
#             feat_right:
#             sample_grid:
#
#         Returns:
#
#         """
#         # note: 对于右特征图，sample_grid要变成负的,
#         feat_right_vol = self.warp(feat_right, -sample_grid)
#         cost_vol = self.match_feat(feat_left, feat_right_vol)
#         return CostVolume(cost_vol.contiguous(), sample_grid)


class DymCostVolBuilder(BaseDymCtVolBuilder):
    _default_match_feat_cfg = dict(
        type='CorrMatchFeat',
        reduce=True,
        keep_dim=False
    )

    def __init__(self, match_feat_cfg: Optional[Dict] = None, sample_mode='bilinear',
                 padding_mode='border'):
        super().__init__()
        self.warp = WarpOpt3D(mode=sample_mode,
                              padding_mode=padding_mode,
                              use_mask=False)
        if match_feat_cfg is None:
            match_feat_cfg = self._default_match_feat_cfg
        self.match_feat = MATCH_FEAT.build(match_feat_cfg)

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor,
                sample_grid: torch.Tensor) -> CostVolume:
        """ dynamic build a cost volume according to sample grid

        Args:
            feat_left:
            feat_right:
            sample_grid:

        Returns:

        """
        # note: as for right feature warp to left, use negative sample grid
        feat_right_vol = self.warp(feat_right, -sample_grid)
        cost_vol = self.match_feat(feat_left, feat_right_vol)
        return CostVolume(cost_vol.contiguous(), sample_grid)


class UniformCostVolBuilder(DymCostVolBuilder):

    def __init__(self, sample_range, sample_num, clip_max, clip_min,
                 match_feat_cfg: Optional[Dict] = None, sample_mode='bilinear'):
        super().__init__(match_feat_cfg, sample_mode)
        self.match_sampler = UniformMatchSampler(sample_range, sample_num, clip_max, clip_min)

    @multimethod
    def forward(self, feat_left: torch.Tensor,
                feat_right: torch.Tensor,
                pri_disp: torch.Tensor):
        """

        Args:
            feat_left:
            feat_right:
            pri_disp

        Returns:

        """
        sample_grid = self.match_sampler(pri_disp)
        return super().forward(feat_left, feat_right, sample_grid)


class DymUniformCostVolBuilder(DymCostVolBuilder):

    def __init__(self, sample_num, clip_max, clip_min,
                 init_alpha=1.0, init_beta=0.0,
                 match_feat_cfg: Optional[Dict] = None, sample_mode='bilinear',
                 freeze=False):
        super().__init__(match_feat_cfg, sample_mode)
        self.match_sampler = DymUniformMatchSampler(sample_num, clip_max, clip_min,
                                                    init_alpha, init_beta)
        if freeze:
            self.match_sampler.eval()

    @multimethod
    def forward(self, feat_left: torch.Tensor,
                feat_right: torch.Tensor,
                pri_disp: torch.Tensor,
                variance: torch.Tensor):
        """

        Args:
            feat_left:
            feat_right:
            pri_disp:
            variance

        Returns:

        """
        sample_grid = self.match_sampler(pri_disp, variance)
        return self._forward(feat_left, feat_right, sample_grid)


class CostVolLookUp(BaseCtVolLookup):
    def __init__(self, sample_range, sample_num, clip_max, clip_min,
                 mode: str = 'bilinear',
                 padding_mode: str = 'border',
                 use_mask: bool = False,
                 volume_format='DHW'):
        super().__init__()
        self.global_cost_vol = None
        self.match_sampler = UniformMatchSampler(sample_range, sample_num, clip_max, clip_min)
        self.lookup_opt = LookUpOpt(mode, padding_mode,
                                    use_mask,
                                    volume_format)

    def retrieve(self, cost_vol):
        self.global_cost_vol = cost_vol

    def forward(self, pri_disp) -> CostVolume:
        """

        Args:
        Returns:

        """
        sample_grid = self.match_sampler(pri_disp)
        if self.global_cost_vol is None:
            raise ValueError("Get None type 'global_cost_vol', please retrieve cost volume before lookup")

        local_cv = self.lookup_opt(self.global_cost_vol, sample_grid)
        return CostVolume(local_cv.contiguous(), sample_grid)

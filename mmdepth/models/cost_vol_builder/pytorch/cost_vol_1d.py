from typing import Generator, Any, List, Optional, Dict
import torch

from mmcv.cnn import build_activation_layer
from mmengine.model import BaseModule

from ..base import BaseCtVolBuilder
from .match_feat import CorrMatchFeat, GroupCorrMatchFeat
from .registry import MATCH_FEAT
from .dym_cost_vol_1d import _BaseDymCtVolBuilder
from mmdepth.models.modules.opts import GatherOpt2D

from mmdepth.models import CostVolume


class CorrCostVolBuilder(BaseCtVolBuilder):
    def __init__(self, in_channels, disp_max=None, disp_min=0, step=1, act_cfg=None):
        super().__init__(in_channels, disp_max, disp_min, step, act_cfg)
        self.match_feat = CorrMatchFeat(reduce=True, keep_dim=False)
        if act_cfg is not None:
            self.act_fn = build_activation_layer(act_cfg)
        else:
            self.act_fn = None

    @property
    def out_channels(self):
        if self.match_feat.out_channels is not None:
            return self.match_feat.out_channels
        else:
            return (self.disp_max - self.disp_min) // self.step

    def forward(self, feat_left, feat_right) -> CostVolume:

        bs, ch, h, w = feat_left.size()
        disp_max = self.disp_max
        disp_min = self.disp_min
        step = self.step

        if disp_max is None:
            disp_max = w // 4  #

        assert disp_max > disp_min, (f"disp_max must be greater than disp_min, "
                                     f"but got disp_max:{disp_max} and disp_min:{disp_min}")

        disp_range = range(disp_min, disp_max, step)
        num_disp = len(disp_range)

        cost_v = feat_left.new_zeros((bs, num_disp, h, w))

        for i, d in enumerate(disp_range):
            if d > 0:
                cost_v[:, i, :, d:] = self.match_feat(feat_left[:, :, :, d:],
                                                      feat_right[:, :, :, :-d])
            else:
                cost_v[:, i, :, :] = self.match_feat(feat_left, feat_right)

        if self.act_fn is not None:
            cost_v = self.act_fn(cost_v)

        # [d,] -> [b=1,d,h=1,w=1]
        sample_grid = torch.arange(disp_min, disp_max, step,
                                   dtype=feat_left.dtype, device=feat_left.device)
        sample_grid = sample_grid[None, :, None, None]

        return CostVolume(cost_v.contiguous(), sample_grid)

    def __repr__(self):
        return 'pytorch backend: corr_cost_volume'


class GwcCostVolBuilder(BaseCtVolBuilder):
    def __init__(self, disp_max=None, disp_min=0, step=1, group_num=4, act_cfg=None):
        super().__init__(disp_max, disp_min, step, act_cfg)
        self.match_feat = GroupCorrMatchFeat(group_num=group_num)

    @property
    def out_channels(self):
        return self.match_feat.out_channels

    def forward(self, feat_left, feat_right) -> CostVolume:
        bs, ch, h, w = feat_left.size()
        disp_max = self.disp_max
        disp_min = self.disp_min
        step = self.step
        group_num = self.match_feat.group_num

        if disp_max is None:
            disp_max = w // 4  #

        assert disp_max > disp_min, (f"disp_max must be greater than disp_min, "
                                     f"but got disp_max:{disp_max} and disp_min:{disp_min}")

        disp_range = range(disp_min, disp_max, step)
        num_disp = len(disp_range)

        cost_v = feat_left.new_zeros((bs, group_num, num_disp, h, w))

        for i, d in enumerate(disp_range):
            if d > 0:
                cost_v[:, :, i, :, d:] = self.match_feat(feat_left[:, :, :, d:],
                                                         feat_right[:, :, :, :-d])
            else:
                cost_v[:, :, i, :, :] = self.match_feat(feat_left, feat_right)

        if self.act_cfg is not None:
            act_fn = build_activation_layer(self.act_cfg)
            cost_v = act_fn(cost_v)

        # [d,] -> [b=1,d,h=1,w=1]
        sample_grid = torch.arange(disp_min, disp_max, step,
                                   dtype=feat_left.dtype, device=feat_left.device)
        sample_grid = sample_grid[None, :, None, None]

        return CostVolume(cost_v.contiguous(), sample_grid)


class ConcatCostVolBuilder(BaseCtVolBuilder):
    def __init__(self, in_channels, disp_max=None, disp_min=0, step=1, act_cfg=None):
        super().__init__(in_channels, disp_max, disp_min, step, act_cfg)
        # self.match_feat = ConcatMatchFeat()
        self._out_channels = in_channels * 2

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, feat_left, feat_right) -> CostVolume:
        """

        Args:
            feat_left:
            feat_right:
        Returns:

        """
        bs, ch, h, w = feat_left.size()
        disp_max = self.disp_max
        disp_min = self.disp_min
        step = self.step

        if disp_max is None:
            disp_max = w // 4  #

        assert disp_max > disp_min, (f"disp_max must be greater than disp_min, "
                                     f"but got disp_max:{disp_max} and disp_min:{disp_min}")

        disp_range = range(disp_min, disp_max, step)
        num_disp = len(disp_range)

        cost_v = feat_left.new_zeros((bs, 2 * ch, num_disp, h, w))

        # fix: not efficient
        # for i, d in enumerate(disp_range):
        #     if d > 0:
        #         cost_v[:, :, i, :, d:] = self.match_feat(feat_left[:, :, :, d:],
        #                                                  feat_right[:, :, :, :-d])
        #     else:
        #         cost_v[:, :, i, :, :] = self.match_feat(feat_left, feat_right)

        for i, d in enumerate(disp_range):
            if d > 0:
                cost_v[:, :ch, i, :, d:] = feat_left[:, :, :, d:]
                cost_v[:, ch:, i, :, d:] = feat_right[:, :, :, :-d]
            else:
                cost_v[:, :ch, i, :, :] = feat_left
                cost_v[:, ch:, i, :, :] = feat_right

        if self.act_cfg is not None:
            act_fn = build_activation_layer(self.act_cfg)
            cost_v = act_fn(cost_v)

        # [d,] -> [b=1,d,h=1,w=1]
        sample_grid = torch.arange(disp_min, disp_max, step,
                                   dtype=feat_left.dtype, device=feat_left.device)
        sample_grid = sample_grid[None, :, None, None]

        return CostVolume(cost_v.contiguous(), sample_grid)

    def __repr__(self):
        return 'pytorch concat'


class CostVolBuilder(BaseCtVolBuilder, _BaseDymCtVolBuilder):
    """ 使用sample grid的方法构建cost volume, 比较消耗内存
    """

    def __init__(self, in_channels, disp_max=None, disp_min=0, step=1, act_cfg=None,
                 match_feat_cfg=None, sample_mode='bilinear', padding_mode='border'):
        BaseCtVolBuilder.__init__(self, in_channels, disp_max=disp_max, disp_min=disp_min,
                                  step=step, act_cfg=act_cfg)
        _BaseDymCtVolBuilder.__init__(self, match_feat_cfg=match_feat_cfg,
                                      sample_mode=sample_mode, padding_mode=padding_mode)
        # self._out_channels =

    @property
    def out_channels(self):
        if self.match_feat.out_channels is not None:
            return self.match_feat.out_channels
        else:
            return (self.disp_max - self.disp_min) // self.step

    def forward(self, feat_left, feat_right) -> CostVolume:
        """

        Args:
            feat_left:
            feat_right:

        Returns:

        """
        bs, ch, h, w = feat_left.size()
        disp_max = self.disp_max
        disp_min = self.disp_min
        step = self.step

        if disp_max is None:
            disp_max = w // 4  #

        assert disp_max > disp_min, (f"disp_max must be greater than disp_min, "
                                     f"but got disp_max:{disp_max} and disp_min:{disp_min}")

        # [d,]
        sample_grid = torch.arange(disp_min, disp_max, step=step,
                                   dtype=feat_left.dtype,
                                   device=feat_left.device)
        # [d,] -> [b=1,d,h=1,w=1] -> [b,d,h,w]
        sample_grid = sample_grid[None, :, None, None]
        sample_grid = sample_grid.expand(bs, -1, h, w)
        #
        return _BaseDymCtVolBuilder._forward(self, feat_left, feat_right, sample_grid)


class CostVolSliceYield(BaseModule):
    """A module that generates cost volume slices for stereo matching.

    This module takes left and right feature maps as input and yields cost volume slices
    along the disparity dimension. It implements a memory-efficient way to generate
    cost volume by using Python generator.

    Args:
        disp_max (int, optional): Maximum disparity. If None, will be set to width//4.
        disp_min (int, optional): Minimum disparity. Defaults to 0.
        step (int, optional): Step size for disparity sampling. Defaults to 1.
        match_feat_cfg (Dict): Config for feature matching module.
            Defaults to dict(type='ConcatMatchFeat').
    """

    def __init__(self,
                 disp_max: Optional[int] = None,
                 disp_min: int = 0,
                 step: int = 1,
                 match_feat_cfg: Dict[str, Any] = dict(type='SquareDiffMatchFeat')):
        super().__init__()

        # Initialize parameters
        self.disp_max = disp_max
        self.disp_min = disp_min
        self.step = step

        # Build feature matching module
        self.match_feat = MATCH_FEAT.build(match_feat_cfg)
        self.gather_opt = GatherOpt2D(dim='w')

        # Placeholders for feature maps
        self._feat_left: Optional[torch.Tensor] = None
        self._feat_right: Optional[torch.Tensor] = None

    @property
    def out_channels(self):
        return self.match_feat.out_channels

    def retrieve(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> None:
        """Store left and right feature maps for later use.

        Args:
            feat_left (torch.Tensor): [b,c,h,w]
            feat_right (torch.Tensor): [b,c,h,w]
        """
        # Verify input tensor dimensions
        assert feat_left.dim() == 4, f"Expected 4D tensor, got {feat_left.dim()}D"
        assert feat_right.dim() == 4, f"Expected 4D tensor, got {feat_right.dim()}D"

        # Verify matching dimensions
        assert feat_left.size() == feat_right.size(), \
            f"Feature size mismatch: {feat_left.size()} vs {feat_right.size()}"

        self._feat_left = feat_left
        self._feat_right = feat_right

    # @property
    # def sample_grid(self):
    #     return self._sample_grid

    def forward(self):
        """Generate cost volume slices using gather with border padding."""
        assert self._feat_left is not None and self._feat_right is not None

        feat_left = self._feat_left
        feat_right = self._feat_right

        B, C, H, W = feat_left.size()
        disp_max = self.disp_max
        if disp_max is None:
            disp_max = W // 4  #

        for d in range(self.disp_min, disp_max, self.step):
            if d > 0:
                shifted_right = self.gather_opt(feat_right, d)
                yield self.match_feat(feat_left, shifted_right), d
            else:
                yield self.match_feat(feat_left, feat_right), d

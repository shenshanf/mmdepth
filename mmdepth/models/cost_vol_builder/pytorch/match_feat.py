from typing import Tuple
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from mmengine.model import BaseModule, Sequential
from mmcv.cnn import ConvModule
from .registry import MATCH_FEAT


class BaseMatchFeat(BaseModule, ABC):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """

        Args:
            feat_left: [b, c, h, w]
            feat_right: [b, c, h, w] or [b, c, d, h, w]

        Returns:

        """
        pass

    @abstractmethod
    @property
    def out_channels(self) -> int:
        """
        cost dim if reduce is True
        Returns:

        """
        pass

    @staticmethod
    def _align_shape(feat_left: torch.Tensor, feat_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            feat_left:
            feat_right:

        Returns:

        """
        # check shape
        assert feat_left.ndim == 4, f"left feature must be a 4D tensor(b,c,h,d), but got {feat_left.shape}"
        assert feat_right.ndim == 4 or feat_right.ndim == 5, \
            f"right feature must be a 4D tensor(b,c,h,d) or 5D tensor(b,c,d,h,w), but got {feat_right.shape}"

        # unsqueeze dim at D dimension, enable broadcast operation
        if feat_right.ndim == 5:
            feat_left = feat_left.unsqueeze(dim=2).expand_as(feat_right)
        return feat_left, feat_right


@MATCH_FEAT.register_module()
class DiffMatchFeat(BaseMatchFeat):
    def __init__(self, in_channels, is_abs: bool = True, reduce: bool = True, keep_dim: bool = False):
        super().__init__(in_channels)
        self.is_abs = is_abs
        self.reduce = reduce
        self.keep_dim = keep_dim

    @property
    def out_channels(self):
        if self.reduce:
            if not self.keep_dim:
                return 1
            else:
                return -1  # means squeezed
        else:
            return self.in_channels

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """

        Args:
            feat_left: [b, c, h, w]
            feat_right: [b, c, h, w] or [b, c, d, h, w]

        Returns:
            cost: reduce and keep_dim [b,1,h,w] or [b,1,d,h,w]
                  reduce and not keep_dim [b,h,w] or [b,d,h,w]
                  not reduce [b,c,h,w] or [b,c,d,h,w]
        """
        # align shape
        feat_left, feat_right = self._align_shape(feat_left, feat_right)

        # compute cost: [b,c,h,w] or [b,c,d,h,w]
        if self.is_abs:
            cost = torch.abs(feat_left - feat_right)
        else:
            cost = feat_left - feat_right

        # 　reduce at channel dimension and normalize
        if self.reduce:
            cost = cost.mean(dim=1, keepdim=self.keep_dim)

        return cost


@MATCH_FEAT.register_module()
class SquareDiffMatchFeat(BaseMatchFeat):
    def __init__(self, in_channels, is_negative=True, reduce: bool = True, keep_dim: bool = False):
        super().__init__(in_channels)
        self.is_negative = is_negative
        self.reduce = reduce
        self.keep_dim = keep_dim

    @property
    def out_channels(self):
        if self.reduce:
            if not self.keep_dim:
                return 1
            else:
                return None  # means squeezed
        else:
            return self.in_channels

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """

        Args:
            feat_left: [b, c, h, w]
            feat_right: [b, c, h, w] or [b, c, d, h, w]

        Returns:
            cost: reduce and keep_dim [b,1,h,w] or [b,1,d,h,w]
                  reduce and not keep_dim [b,h,w] or [b,d,h,w]
                  not reduce [b,c,h,w] or [b,c,d,h,w]
        """
        # align shape
        feat_left, feat_right = self._align_shape(feat_left, feat_right)

        # compute cost: [b,c,h,w] or [b,c,d,h,w]
        cost = (feat_left - feat_right) ** 2

        # 　reduce at channel dimension and normalize
        if self.reduce:
            cost = cost.mean(dim=1, keepdim=self.keep_dim)

        cost = -cost if self.is_negative else cost

        return cost


@MATCH_FEAT.register_module()
class CorrMatchFeat(BaseMatchFeat):
    def __init__(self, in_channels, reduce: bool = True, keep_dim: bool = False):
        super().__init__(in_channels)
        self.reduce = reduce
        self.keep_dim = keep_dim

    @property
    def out_channels(self):
        if self.reduce:
            if not self.keep_dim:
                return 1
            else:
                return None # means squeezed
        else:
            return self.in_channels

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """

        Args:
            feat_left: [b, c, h, w]
            feat_right: [b, c, h, w] or [b, c, d, h, w]

        Returns:
            cost: reduce and keep_dim [b,1,h,w] or [b,1,d,h,w]
                  reduce and not keep_dim [b,h,w] or [b,d,h,w]
                  not reduce [b,c,h,w] or [b,c,d,h,w]
        """
        # align shape
        feat_left, feat_right = self._align_shape(feat_left, feat_right)

        # compute cost: [b,c,h,w] or [b,c,d,h,w]
        cost = feat_left * feat_right

        # 　reduce at channel dimension and normalize
        if self.reduce:
            cost = cost.mean(dim=1, keepdim=self.keep_dim)
        return cost


@MATCH_FEAT.register_module()
class GroupCorrMatchFeat(BaseMatchFeat):
    def __init__(self, in_channels, group_num=4):
        super().__init__(in_channels)
        self.group_num = group_num

    @property
    def out_channels(self):
        return self.group_num

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """

        Args:
            feat_left: [b, c, h, w]
            feat_right: [b, c, h, w] or [b, c, d, h, w]

        Returns:

        """
        feat_left, feat_right = self._align_shape(feat_left, feat_right)

        bs, ch = feat_left.shape[:2]
        h, w = feat_left.shape[-2:]

        assert ch % self.group_num == 0, \
            f"feature channel:{ch} can't be divisible by group_nums:{self.group_num}"
        ch_per_group = ch // self.group_num

        # compute cost: [b,c,h,w] or [b,c,d,h,w]
        cost = feat_left * feat_right

        # split channel by group
        if cost.ndim == 4:
            # [b, g, c1, h, w]
            cost = cost.view(bs, self.group_num, ch_per_group, h, w)
        else:
            # [b, g, c1, d, h, w]
            cost = cost.view(bs, self.group_num, ch_per_group, -1, h, w)

        # reduce per group
        # [b, g, h, w] or [b, g, d, h, w]
        cost = cost.mean(dim=2, keepdim=False)
        return cost


@MATCH_FEAT.register_module()
class ConcatMatchFeat(BaseMatchFeat):
    def __init__(self, in_channels):
        super().__init__(in_channels)

    @property
    def out_channels(self) -> int:
        return self.in_channels * 2

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """

        Args:
            feat_left: [b, c, h, w]
            feat_right: [b, c, h, w] or [b, c, d, h, w]

        Returns:

        """
        feat_left, feat_right = self._align_shape(feat_left, feat_right)
        cost = torch.concat((feat_left, feat_right), dim=1)
        return cost


class VarianceMatchFeat(BaseMatchFeat):
    # todo: variance based cost volume

    @property
    def out_channels(self):
        return

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        ...


@MATCH_FEAT.register_module()
class PDSMatchFeat(BaseMatchFeat):
    """
    see: "Practical Deep Stereo (PDS): Toward applications-friendly deep stereo matching", NeurIPS 2018
    use a convolutional layer to squeeze the channel dimension
    """

    def __init__(self, in_channels, h_channels, out_channels):
        """

        Args:
            in_channels:
            out_channels:
        """
        super().__init__(in_channels)
        # official code: conv1 and conv3: kernel_size=3, padding=1
        self.conv1 = nn.Conv2d(in_channels * 2, h_channels, kernel_size=1, padding=0)
        layer2a = ConvModule(h_channels, h_channels, kernel_size=3, padding=1,
                             act_cfg=dict(type='LeakyReLU', negative_slope=0.1, inplace=True),
                             norm_cfg=dict(type='InstanceNorm2d', num_features=h_channels, affine=True))
        layer2b = ConvModule(h_channels, h_channels, kernel_size=3, padding=1,
                             act_cfg=dict(type='LeakyReLU', negative_slope=0.1, inplace=True),
                             norm_cfg=dict(type='InstanceNorm2d', num_features=h_channels, affine=True))
        self.conv2 = Sequential(layer2a, layer2b)
        self.conv3 = nn.Conv2d(h_channels, out_channels, kernel_size=1, padding=0)
        self._out_channels = out_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, feat_left: torch.Tensor, feat_right: torch.Tensor) -> torch.Tensor:
        """

        Args:
            feat_left: [b,c,h,w]
            feat_right: [b,c,h,w] or [b,c,d,h,w]

        Returns:
                [b,c1,h,w]
        """
        # align shape
        feat_left, feat_right = self._align_shape(feat_left, feat_right)

        x = torch.cat([feat_left, feat_right], dim=1)
        x = self.conv1(x)

        identity = x
        x = self.conv2(x)
        return self.conv3(identity + x)  # residual connect

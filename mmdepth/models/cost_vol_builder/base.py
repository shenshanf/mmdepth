from typing import Union, Optional, Sequence
from abc import ABC, abstractmethod
from mmengine.model import BaseModule
import torch

from mmdepth.models.structures import CostVolume


class BaseCtVolBuilder(BaseModule, ABC):
    def __init__(self, in_channels, disp_max=None, disp_min=0, step=1, act_cfg=None):
        super().__init__()
        self.in_channels = in_channels
        self.disp_max = disp_max
        self.disp_min = disp_min
        self.step = step
        self.act_cfg = act_cfg

    @abstractmethod
    @property
    def out_channels(self):
        pass

    @abstractmethod
    def forward(self, feat_left, feat_right) -> CostVolume:
        """

        Args:
            feat_left:
            feat_right:

        Returns:

        """
        ...


class BaseDymCtVolBuilder(BaseModule, ABC):
    ...


class BaseCtVolLookup(BaseModule, ABC):
    ...

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union
from multimethod import multimethod
import torch
import torch.nn as nn

from mmengine.model import BaseModule


class _UniformMatchSampler(BaseModule):
    def __init__(self, sample_num, clip_max, clip_min=0):
        super().__init__()
        assert sample_num > 1, "sample_num must be greater than 1"
        assert clip_max > clip_min, "clip_max must be greater than clip_min"
        self.sample_num = sample_num
        self.clip_max = clip_max
        self.clip_min = clip_min

    def forward(self, sample_range: torch.Tensor):
        """

        Args:
            sample_range: [b, h, w, 2]

        Returns:

        """
        assert sample_range.size(1) == 2, "sample_range should have shape [b, 2, h, w]"
        sample_num = self.sample_num
        # _, _, H, W = search_range.size()
        s_min = sample_range[:, :1, :, :]
        s_max = sample_range[:, 1:, :, :]
        # assert torch.all(s_max >= s_min), "max values should be >= min values"

        # note：保证最小的step为1
        modify_range = (sample_num - (s_max - s_min + 1)).clamp(min=0) / 2.0
        s_min = (s_min - modify_range).clamp(min=self.clip_min, max=self.clip_max)
        s_max = (s_max + modify_range).clamp(min=self.clip_min, max=self.clip_max)

        # note：均匀采样，采样点包括 min max端点
        sample_step = (s_max - s_min) / (sample_num - 1)  # [N,1,H,W]
        sample_step = sample_step.repeat([1, sample_num, 1, 1])  # [N,1,H,W] -> [N,S,H,W]
        sample_grid = torch.linspace(0, sample_num - 1, sample_num,
                                     device=sample_range.device, dtype=sample_range.dtype)  # [S] 这里出来是没有step信息的
        sample_grid = sample_grid[None, :, None, None]  # [S] -> [N=1,S,H=1,W=1]
        # [N=1,S,H=1,W=1] * [N,S,H,W] = [N,S,H,W] broadcast at N,H,W dimension
        sample_grid = sample_grid * sample_step
        # [N,1,H,W] + [N,S,H,W] broadcast at S dimension
        sample_grid = s_min + sample_grid

        return sample_grid.contiguous()


class UniformMatchSampler(_UniformMatchSampler):
    def __init__(self, sample_range, sample_num, clip_max, clip_min=0):
        super().__init__(sample_num, clip_max, clip_min)
        self.sample_range = sample_range

    def _convert(self, device: torch.device):
        sample_range = self.sample_range
        assert len(sample_range) == 2
        assert sample_range[0] < sample_range[1], \
            f"search_min: {sample_range[0]} must less than search_max: {sample_range[1]}"
        sample_range = torch.Tensor(sample_range, device=device)
        # [2] -> [B=1,2,H=1,W=1]
        sample_range = sample_range[None, :, None, None]
        return sample_range

    @multimethod
    def forward(self, pri_disp: torch.Tensor):
        """

        Args:
            pri_disp: [b,1,h,w]

        Returns:

        """
        sample_range = self._convert(pri_disp.device)
        # [b,2,h,w] <- [b,c=1,h,w] + [b,2,h,w]
        sample_range = pri_disp + sample_range  #
        return super().forward(sample_range)


class DymUniformMatchSampler(_UniformMatchSampler):
    def __init__(self, sample_num, clip_max, clip_min=0,
                 init_alpha=1.0, init_beta=0.0):
        super().__init__(sample_num, clip_max, clip_min)
        self.alpha = nn.Parameter(torch.full((1,),
                                             fill_value=init_alpha))
        self.beta = nn.Parameter(torch.full((1,),
                                            fill_value=init_beta))

    @multimethod
    def forward(self, pri_disp: torch.Tensor, variance: torch.tensor):
        """

        Args:
            pri_disp: [b,1,h,w]
            variance:
        Returns:

        """
        sample_range = torch.cat([self.alpha * variance + self.beta,
                                  -self.alpha * variance + self.beta], dim=1)
        sample_range = pri_disp + sample_range
        return super().forward(sample_range)


# class
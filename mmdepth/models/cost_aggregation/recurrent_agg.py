from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList
from ..cost_vol_builder import CostVolSliceYield

from ..modules import StackedConvGRUCell

from mmdepth.registry import AGGREGATORS


@AGGREGATORS.register_module()
class RecurrentAggregator(BaseModule):
    """ Recurrent MVS aggregation module
    """

    def __init__(self, in_channels, h_channels,
                 disp_max: Optional[int] = None,
                 disp_min: int = 0,
                 step: int = 1,
                 match_feat_cfg: Dict[str, Any] = dict(type='SquareDiffMatchFeat',
                                                       reduction=None,
                                                       is_negative=True),
                 num_gru_cells=3,
                 kernel_size=3,
                 cost_yield_backend='pytorch'):
        super().__init__()
        self.cost_vol_slice_yield = CostVolSliceYield(disp_max, disp_min, step,
                                                      match_feat_cfg, backend=cost_yield_backend)

        self.stacked_grus = StackedConvGRUCell(in_channels, h_channels,
                                               kernel_size, bias=True,
                                               num_stacked=num_gru_cells)

        self.final_layer = nn.Conv2d(h_channels, 1, 3, 1, 1)

    def forward(self, feat_left, feat_right):
        """

        Args:
            feat_left:
            feat_right:

        Returns:

        """
        self.cost_vol_slice_yield.retrieve(feat_left, feat_right)
        hidden_states = None  #
        results = []
        for vol_slice, d in self.cost_vol_slice_yield():
            hidden_states = self.stacked_grus(hidden_states, vol_slice)
            results.append(self.final_layer(hidden_states[-1]))
        # d*[b,c,h,w] -> [b,c,d,h,w] or d*[b,h,w] -> [b,d,h,w]
        return torch.stack(results, dim=-3).contiguous()

from typing import Optional, Tuple, List

import torch
from mmcv.cnn import ConvModule, build_activation_layer
from mmengine.model import BaseModule, ModuleList
from torch import nn as nn

from mmdepth.models.modules.bricks import BaseUnetDecoder
from mmdepth.registry import AGGREGATORS


class PSMEncoder(BaseModule):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1, dilation=1,
                 norm_cfg=dict(type='BatchNorm3d'),
                 act_cfg=dict(type='ReLu')):
        super().__init__()
        self.down_conv = ConvModule(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, bias='auto',
                                    conv_cfg=dict(type='Conv3d'),
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg)
        self.layer = ConvModule(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, dilation=1, bias='auto',
                                conv_cfg=dict(type='Conv3d'),
                                norm_cfg=norm_cfg,
                                act_cfg=None)
        self.act = build_activation_layer(act_cfg)

    def forward(self, x, skip_x=None):
        x = self.down_conv(x)
        x = self.layer(x)
        if skip_x:
            return self.act(x + skip_x)
        else:
            return self.act(x)


class PSMDecoder(BaseUnetDecoder):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1, dilation=1,
                 norm_cfg=dict(type='BatchNorm3d'),
                 act_cfg=dict(type='ReLu', inplace=True)):
        first_block = ConvModule(in_channels, out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, dilation=dilation, bias='auto',
                                 conv_cfg=dict(type='ConvTranspose3d'),
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)
        fusion = self.add_fusion

        final_block = build_activation_layer(act_cfg) if act_cfg is not None else None
        super().__init__(first_block, fusion, skip_block=None, final_block=final_block)


@AGGREGATORS.register_module()
class PSMHourGlass(BaseModule):
    """HourGlass Network implemented for 'PSMNet'.

    A U-Net like architecture with skip connections and optional intermediate outputs.

    """

    def __init__(self, channels, norm_cfg, act_cfg):
        super().__init__()
        assert len(channels) == 3
        # Encoder path (down sampling)
        self.encoder0 = PSMEncoder(channels[0], channels[1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.encoder1 = PSMEncoder(channels[1], channels[2], norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Bottleneck
        self.middle_block = nn.Identity()

        # Decoder path (up sampling)
        self.decoder0 = PSMDecoder(channels[2], channels[1], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.decoder1 = PSMDecoder(channels[1], channels[2][0], norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor,
                pre_skip: Optional[torch.Tensor] = None,
                post_skip: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """Forward pass of the HourGlass network.

        Args:
            x (torch.Tensor): Input tensor
            pre_skip (torch.Tensor, optional): Skip connection from encoder path
            post_skip (torch.Tensor, optional): Skip connection for additional features

        Returns:
            tuple[torch.Tensor]: Tuple of output tensors. Always includes final output,
                optionally includes intermediate features based on pre_out and post_out flags
        """
        # encode stage
        x = self.encoder0(x, post_skip)
        pre_feat = x if pre_skip is None else pre_skip
        x = self.encoder1(x)

        # bottleneck
        x = self.middle_block(x)

        # decode stage
        x = self.decoder0(x, pre_feat)
        post_feat = x
        x = self.decoder1(x)

        return x, pre_feat, post_feat


@AGGREGATORS.register_module()
class StackPSMHourGlass(BaseModule):
    """Stack of Hourglass modules that can be configured with different number of stacks.

    Args:
        num_stacks (int): Number of hourglass modules to stack. Default: 3
    """

    def __init__(self, channels, num_stacks: int = 3,
                 norm_cfg=dict(type='BatchNorm3d'),
                 act_cfg=dict(type='ReLu')):
        super(StackPSMHourGlass, self).__init__()

        self.stack_hourglass = ModuleList()
        for idx in range(num_stacks):
            self.stack_hourglass.append(
                PSMHourGlass(channels, norm_cfg, act_cfg))

    def forward(self, cost_vol: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through stacked hourglass modules.

        Args:
            cost_vol (torch.Tensor): Input cost volume

        Returns:
            List[torch.Tensor]: List of aggregated cost volumes from each stack
        """
        agg_cost_vols = []  # store aggregated cost volumes from each stack

        # Process first hourglass module
        # encoder_skip: intermediate features from encoder path, used for skip connections
        # decoder_skip: intermediate features from decoder path, used for skip connections
        agg_cost_vol, encoder_skip, decoder_skip = self.hourglasses[0](cost_vol)
        agg_cost_vols.append(agg_cost_vol + cost_vol)

        # Process remaining hourglass modules
        last_decoder_skip = decoder_skip  # decoder features from previous hourglass
        for i in range(1, self.num_stacks):
            agg_cost_vol, _, new_decoder_skip = self.hourglasses[i](
                agg_cost_vols[-1],  # last aggregated cost volume
                last_decoder_skip,  # decoder features from previous hourglass
                encoder_skip  # encoder features from first hourglass
            )
            agg_cost_vols.append(agg_cost_vol + cost_vol)  # residual connect with first cost volume
            last_decoder_skip = new_decoder_skip

        return agg_cost_vols


@AGGREGATORS.register_module()
class StackX3PSMHourGlass(BaseModule):
    """ StackHourGlass configured with num_stacks=3 for PSMNet"""

    def __init__(self, channels):
        super().__init__()
        self.hourglass1 = PSMHourGlass(channels)
        self.hourglass2 = PSMHourGlass(channels)
        self.hourglass3 = PSMHourGlass(channels)

    def forward(self, cost_v):
        """
        Args:
            cost_v:

        Returns:

        """
        ag_cost_vols = []
        ag_cost_vol, cost_pre_v1, cost_post_v1 = self.hourglass1(cost_v)
        ag_cost_vol = ag_cost_vol + cost_v
        ag_cost_vols.append(ag_cost_vol)

        ag_cost_vol, _, cost_post_v2 = self.hourglass2(ag_cost_vol, cost_post_v1, cost_pre_v1)
        ag_cost_vol = ag_cost_vol + cost_v
        ag_cost_vols.append(ag_cost_vol)

        ag_cost_vol, _, _ = self.hourglass2(ag_cost_vol, cost_post_v2, cost_pre_v1)
        ag_cost_vol = ag_cost_vol + cost_v
        ag_cost_vols.append(ag_cost_vol)

        return ag_cost_vols

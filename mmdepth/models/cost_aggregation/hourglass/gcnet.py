import torch
from mmcv.cnn import ConvModule
from mmengine.model import Sequential, ModuleList
from torch import nn as nn

from mmdepth.models.modules.bricks import BaseUnetDecoder, BaseUNet
from mmdepth.registry import AGGREGATORS


class GCSkipBlock(Sequential):
    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg, stride=2, num_conv=3):
        layers = []
        for idx in range(num_conv):
            _stride = stride if idx == 0 else 1  # first conv for stride 2
            layers.append(
                ConvModule(in_channels, out_channels,
                           kernel_size=3, stride=_stride, padding=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg,
                           conv_cfg=dict(type='Conv3d')))
        super().__init__(layers)


class GCDecoder(BaseUnetDecoder):
    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg, num_skip_conv=3):
        first_block = ConvModule(in_channels, out_channels,
                                 kernel_size=3, stride=2, padding=1,
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg,
                                 conv_cfg=dict(type='ConvTranspose3d'))
        fusion = self.add_fusion
        final_block = nn.Identity()
        skip_block = GCSkipBlock(out_channels, out_channels, norm_cfg, act_cfg, stride=2, num_conv=num_skip_conv)
        super().__init__(first_block=first_block, fusion=fusion,
                         skip_block=skip_block, final_block=final_block)


@AGGREGATORS.register_module()
class GCHourGlass(BaseUNet):
    def __init__(self, channels, norm_cfg, act_cfg, num_stages, num_skip_conv):
        """

        Args:
            channels:
            norm_cfg:
            act_cfg:
            num_stages:
            num_skip_conv:

        """
        assert len(channels) == num_stages + 1
        encoders = ModuleList()
        decoders = ModuleList()

        for idx in range(num_stages):
            # encoders
            encoders.append(
                ConvModule(channels[idx], channels[idx + 1],
                           kernel_size=3, stride=2, padding=1,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg,
                           conv_cfg=dict(type='Conv3d')))
            # decoder
            decoders.append(
                GCDecoder(channels[idx + 1], channels[idx],
                          norm_cfg=norm_cfg, act_cfg=act_cfg,
                          num_skip_conv=num_skip_conv)
            )
        neck_layer = GCSkipBlock(channels[-1], channels[-1],
                                 norm_cfg, act_cfg, stride=2, num_conv=num_skip_conv)

        super().__init__(encoders, decoders, neck_layer,
                         skip_first=True, multi_out=False)

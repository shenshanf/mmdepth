from itertools import accumulate as itertools_accum
from mmengine.model import BaseModule, Sequential, ModuleList
from mmcv.cnn import ConvModule
import torch
from torch import nn as nn

from mmdepth.registry import AGGREGATORS

from .hourglass import GCHourGlass, GCSkipBlock, NLAMHourGlass, StackPSMHourGlass, StackGWCHourGlass
from ..modules import BasicBlock, SpatialInterpol


@AGGREGATORS.register_module()
class StereoNetAggregator(BaseModule):
    """ see paper: StereoNet:
    https://github.com/meteorshowers/X-StereoLab/blob/master/disparity/models/stereonet_disp.py"""

    def __init__(self, in_channels, num_layers=4, kernel_size=3,
                 norm_cfg=dict(type='BatchNorm2d'),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.2, inplace=True)):
        super().__init__()
        self.layers = Sequential()
        for _ in range(num_layers):
            self.layers.append(ConvModule(in_channels, in_channels,
                                          kernel_size, stride=1, padding=kernel_size // 2,
                                          conv_cfg=dict(type='Conv3d'),
                                          norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.final_layer = nn.Conv3d(in_channels, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, cost_vol):
        return self.final_layer(self.layers(cost_vol))


class PSMNetClassifier(BaseModule):
    def __init__(self, base_channels,
                 conv_cfgs=[dict(type='Conv3d'), dict(type='Conv3d')],
                 norm_cfg=dict(type='BatchNorm3d'),
                 act_cfg=dict(type='ReLu')):
        super().__init__()
        self.conv = ConvModule(base_channels, base_channels,
                               kernel_size=3, stride=1, padding=1,
                               conv_cfg=conv_cfgs[0],
                               norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.sq_conv = ConvModule(base_channels, base_channels,
                                  kernel_size=3, stride=1, padding=1,
                                  conv_cfg=conv_cfgs[1],
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.sq_conv(self.conv(x))


@AGGREGATORS.register_module()
class GCNetAggregator(BaseModule):
    def __init__(self, channels, norm_cfg, act_cfg, num_stages=3, num_skip_conv=3):
        """

        Args:
            channels:
            norm_cfg:
            act_cfg:
            num_stages:
            num_skip_conv:
        """
        super().__init__()
        self.hourglass = GCHourGlass(channels, norm_cfg, act_cfg, num_stages, num_skip_conv)
        self.recover_layer = ConvModule(channels[0], channels[0],
                                        kernel_size=3, stride=2, padding=1,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg,
                                        conv_cfg=dict(type='ConvTranspose3d'))
        self.recover_skip = GCSkipBlock(channels[0], channels[0],
                                        norm_cfg, act_cfg, num_conv=num_skip_conv - 1)
        self.classify = ConvModule(channels[0], out_channels=1,
                                   kernel_size=3, stride=2, padding=1,
                                   norm_cfg=norm_cfg,
                                   act_cfg=act_cfg,
                                   conv_cfg=dict(type='ConvTranspose3d'))

    def forward(self, cost_vol):
        identity = cost_vol
        x = self.hourglass(cost_vol)
        x = self.recover_layer(x) + self.recover_skip(identity)
        return self.classify(x)


class NLAMAggregator(BaseModule):
    def __init__(self, channels, num_stage, num_nonlocal, norm_cfg, act_cfg):
        super().__init__()
        self.hourglass = NLAMHourGlass(channels, num_stage, num_nonlocal, norm_cfg, act_cfg)
        # recover layer: recover the spatial dimension and squeeze the channel dimension
        self.recover_layer = Sequential([
            ConvModule(channels[0], channels[0] // 2,
                       kernel_size=1, stride=1, padding=1,
                       conv_cfg=dict(type='Conv3d'),
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(channels[0] // 2, channels[0] // 2,
                       kernel_size=3, stride=2, padding=1,
                       conv_cfg=dict(type='ConvTranspose3d'),
                       norm_cfg=norm_cfg, act_cfg=act_cfg)])
        self.classify = ConvModule(channels[0] // 2, out_channels=1,
                                   kernel_size=1, stride=1, padding=1,
                                   conv_cfg=dict(type='Conv3d'),
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, cost_vol):
        x = self.recover_layer(self.hourglass(cost_vol))
        return self.classify(x)


class PSMFirstBlock(Sequential):
    def __init__(self, in_channels, norm_cfg, act_cfg):
        super().__init__([ConvModule(in_channels, in_channels,
                                     kernel_size=3, stride=1, padding=1,
                                     conv_cfg=dict(type='Conv3d'),
                                     norm_cfg=norm_cfg, act_cfg=act_cfg),
                          ConvModule(in_channels, in_channels,
                                     kernel_size=3, stride=1, padding=1,
                                     conv_cfg=dict(type='Conv3d'),
                                     norm_cfg=norm_cfg, act_cfg=act_cfg),
                          BasicBlock(in_channels, in_channels,
                                     kernel_size=3, stride=1, padding=1,
                                     conv_cfgs=dict(type='Conv3d'),
                                     norm_cfgs=[norm_cfg, None], act_cfgs=[act_cfg, None])])


class PSMNetAggregator(BaseModule):
    def __init__(self, channels, norm_cfg, act_cfg, num_stacks, interpol_size,
                 share_classifier=False, skip_connect_final=True):
        super().__init__()
        self.first_block = PSMFirstBlock(channels[0], norm_cfg, act_cfg)

        self.stack_hourglass = StackPSMHourGlass(channels, num_stacks=num_stacks,
                                                 norm_cfg=norm_cfg, act_cfg=act_cfg)
        if share_classifier:  # share weights
            self.classifiers = PSMNetClassifier(channels[0], channels[0],
                                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            self.classifiers = ModuleList()
            for _ in range(num_stacks):
                self.classifiers.append(PSMNetClassifier(channels[0], channels[0],
                                                         norm_cfg=norm_cfg, act_cfg=act_cfg))

        self.share_classifier = share_classifier
        self.skip_connect_final = skip_connect_final
        self.spatial_interpol = SpatialInterpol(size=interpol_size, mode='trilinear', align_corners=True)

    def forward(self, cost_vol):
        x = self.first_block(cost_vol)
        temp_cost_vols = self.stack_hourglass(x)

        results = []
        # classifier
        if self.share_classifier:
            for temp_cost_vol in temp_cost_vols:
                results.append(self.classifiers(temp_cost_vol))
        else:
            for temp_cost_vol, classifier in zip(temp_cost_vols, self.classifiers):
                results.append(classifier(temp_cost_vol))
        # skip connection
        if self.skip_connect_final:
            itertools_accum(results, torch.add)
        # interpolation
        results = [self.spatial_interpol(v) for v in results]
        return results


class GWCNetAggregator(PSMNetAggregator):
    def __init__(self, channels, norm_cfg, act_cfg, num_stacks, num_stage, interpol_size,
                 share_classifier=False, skip_connect_final=False):
        super().__init__(channels, norm_cfg, act_cfg, num_stacks, interpol_size,
                         share_classifier, skip_connect_final)
        self.stack_hourglass = StackGWCHourGlass(channels, norm_cfg=norm_cfg, act_cfg=act_cfg,
                                                 num_stage=num_stage, num_stack=num_stacks)

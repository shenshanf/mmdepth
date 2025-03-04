from mmcv.cnn import ConvModule
from mmengine.model import ModuleList, BaseModule

from mmdepth.models.modules.bricks import BaseUNet
from .psmnet import PSMDecoder, PSMEncoder
from mmdepth.registry import AGGREGATORS


class GWCDecoder(PSMDecoder):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1, dilation=1,
                 norm_cfg=dict(type='BatchNorm2d'),
                 act_cfg=dict(type='ReLu', inplace=True)):
        super().__init__(in_channels, out_channels,
                         kernel_size, stride, padding, dilation,
                         act_cfg)
        self.skip_block = ConvModule(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=norm_cfg,
            act_cfg=None)


@AGGREGATORS.register_module()
class GWCHourGlass(BaseUNet):
    def __init__(self, channels, norm_cfg, act_cfg, num_stage=2):
        encoders = ModuleList()
        decoders = ModuleList()
        for idx in range(num_stage):
            encoders.append(
                PSMEncoder(channels[idx], channels[idx + 1], norm_cfg=norm_cfg, act_cfg=act_cfg))

            decoders.append(
                GWCDecoder(channels[idx + 1], channels[idx], norm_cfg=norm_cfg, act_cfg=act_cfg)
            )

        super().__init__(encoders, decoders, skip_first=True, multi_out=False)


@AGGREGATORS.register_module()
class StackGWCHourGlass(BaseModule):
    def __init__(self, channels, norm_cfg, act_cfg, num_stage=2, num_stack=3):
        super().__init__()
        self.stack_hourglass = ModuleList()
        for idx in range(num_stack):
            self.stack_hourglass.append(
                GWCHourGlass(channels, norm_cfg, act_cfg, num_stage=num_stage))

    def forward(self, x):
        results = []
        for stack_hourglass in self.stack_hourglass:
            x = stack_hourglass(x)
            results.append(x)
        return results

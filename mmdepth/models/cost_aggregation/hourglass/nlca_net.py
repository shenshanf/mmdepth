from mmcv.cnn.resnet import BasicBlock

from mmengine.model import BaseModule, ModuleList, Sequential
from mmcv.cnn import ConvModule

from mmdepth.registry import AGGREGATORS

from mmdepth.models.modules.bricks import BaseUnetDecoder, BaseUNet, NonLocal3d


class NLAMEncoder(BaseModule):
    def __init__(self, in_channels, out_channels, stride, norm_cfg, act_cfg):
        super().__init__()

        self.first_layer = ConvModule(in_channels, out_channels,
                                      kernel_size=3, stride=stride, padding=1,
                                      conv_cfg=dict(type='Conv3d'),
                                      norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.res_layer = BasicBlock(in_channels, out_channels, stride=1)  # fix

    def forward(self, x):
        return self.res_layer(self.first_layer(x))


class NLAMDecoder(BaseUnetDecoder):
    def __init__(self, in_channels, out_channels, stride, norm_cfg, act_cfg):
        first_block = Sequential([
            ConvModule(in_channels, out_channels,
                       kernel_size=3, stride=stride, padding=1,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg,
                       conv_cfg=dict(type='ConvTranspose3d')),
            BasicBlock(in_channels, out_channels, stride=1)
        ])
        fusion = self.add_fusion
        super().__init__(first_block=first_block, fusion=fusion,
                         skip_block=None, final_block=None)


@AGGREGATORS.register_module()
class NLAMHourGlass(BaseUNet):
    """See paper: 'NLCA-Net: a non-local context attention network for stereo matching'
    """

    def __init__(self, channels, num_stage, num_nonlocal, norm_cfg, act_cfg):
        encoders = ModuleList()
        decoders = ModuleList()

        for idx in range(num_stage):
            stride = 1 if idx == 0 else 2
            encoders.append(
                NLAMEncoder(channels[idx], channels[idx + 1],
                            stride=stride, norm_cfg=norm_cfg, act_cfg=act_cfg))
            if idx > 0:
                decoders.append(
                    NLAMDecoder(channels[idx + 1], channels[idx],
                                stride=2, norm_cfg=norm_cfg, act_cfg=act_cfg))

        # neck layer: non-local 3d * num_nonlocal
        neck_layer = Sequential()
        for _ in range(num_nonlocal):
            neck_layer.append(NonLocal3d(in_channels=channels[-1],
                                         num_heads=1, window_size=-1, reduction=1,
                                         use_scale=True, norm_cfg=norm_cfg, act_cfg=act_cfg,
                                         mode='embedded_gaussian'))

        super().__init__(encoders, decoders, neck_layer,
                         skip_first=False, multi_out=False)

    #

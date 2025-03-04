import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Optional, Sequence, Tuple, Union, List

from mmengine.model import BaseModule, ModuleList
from mmengine.registry import MODELS
from mmdepth.registry import PREDICT_HEADERS
from mmdepth.models.modules.bricks import BaseUNet, BaseUnetDecoder

from mmcv.cnn import ConvModule


class BasicBlock(BaseModule):
    """Basic decoder block for DispNetC stereo matching network.

    This block performs upsampling of features, disparity prediction, and feature fusion.
    For each block, it includes:
    1. Deconvolution for feature upsampling
    2. Current-scale disparity prediction
    3. Disparity upsampling
    4. Feature fusion with skip connection (optional)
    """

    def __init__(self, in_channels, out_channels, inter_channels, skip_channels=None,
                 act_cfgs=dict(deconv=dict(type='LeakyReLu', negative_slope=0.1)),
                 norm_cfgs=dict(deconv=None)):
        """Initialize BasicBlock.

        Args:
            in_channels (int): Input channel number
            out_channels (int): Output channel number
            inter_channels (int): Intermediate channel number after deconv
            skip_channels (int, optional): Channel number of skip connection. Defaults to None.
            act_cfgs (dict): Activation function configurations for different layers
            norm_cfgs (dict): Normalization configurations for different layers
        """
        super().__init__()

        # Deconvolution layer for feature upsampling
        self.de_conv = ConvModule(
            in_channels, inter_channels,
            kernel_size=3, stride=2, padding=1,
            conv_cfg=dict(type='DeConv2d'),
            act_cfg=act_cfgs.get('deconv'),
            norm_cfg=norm_cfgs.get('deconv')
        )

        # Determine fusion channels based on skip connection
        if skip_channels is None:
            self.skip = False
            iconv_channels = inter_channels + 1  # +1 for upsampled disparity
        else:
            self.skip = True
            iconv_channels = inter_channels + skip_channels + 1

        # Fusion convolution layer
        self.iconv = ConvModule(
            iconv_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            conv_cfg=dict(type='Conv2d'),
            act_cfg=act_cfgs.get('iconv'),
            norm_cfg=norm_cfgs.get('iconv')
        )

        # Disparity prediction layer
        self.pred = ConvModule(
            in_channels, out_channels=1,
            kernel_size=3, stride=1, padding=1,
            conv_cfg=dict(type='Conv2d'),
            act_cfg=act_cfgs.get('pred'),
            norm_cfg=norm_cfgs.get('pred')
        )

        # Disparity upsampling layer
        self.up_sample = ConvModule(
            in_channels=1, out_channels=1,
            kernel_size=3, stride=2, padding=1,
            conv_cfg=dict(type='DeConv2d'),
            act_cfg=act_cfgs.get('upsample'),
            norm_cfg=norm_cfgs.get('upsample')
        )

    def forward(self, x, skip=None):
        """Forward function.

        Args:
            x (Tensor): Input feature tensor
            skip (Tensor, optional): Skip connection tensor. Defaults to None.

        Returns:
            tuple:
                - Tensor: Fused feature map
                - Tensor: Predicted disparity map at current scale
        """
        identity = x

        # 1. Feature upsampling through deconvolution
        x = self.de_conv(x)

        # 2. Predict disparity at current scale
        pred_disp = self.pred(identity)

        # 3. Upsample predicted disparity
        up_pred_disp = self.up_sample(pred_disp)

        # 4. Feature fusion
        if self.skip:
            assert skip is not None
            x = torch.cat([x, skip, up_pred_disp], dim=1)
        else:
            assert skip is None, "Skip connection tensor is required when skip_channels set 'None'"
            x = torch.cat([x, up_pred_disp], dim=1)

        # Final convolution for feature fusion
        x = self.iconv(x)

        return x, pred_disp


@PREDICT_HEADERS.register_module()
class DispNetDecoder(BaseModule):
    """Prediction head of DispNetC for stereo matching.

    This module implements a multiscale disparity prediction decoder that consists of:
    1. Multiple BasicBlocks for hierarchical feature decoding and disparity estimation
    2. A final disparity prediction layer at the finest scale

    Args:
        channels (list[int]): Channel numbers for each level
        skip_channels (list[int], optional): Skip connection channel numbers
        num_blocks (int): Number of BasicBlocks used for decoding. Defaults to 5
        act_cfgs (dict): Activation configurations for different layers
        norm_cfgs (dict): Normalization configurations for different layers
    """

    def __init__(self, channels, skip_channels, num_blocks=5,
                 act_cfgs=dict(deconv=dict(type='LeakyReLu', negative_slope=0.1)),
                 norm_cfgs=dict(deconv=None)):
        super().__init__()

        # Validate input parameters
        assert len(channels) == num_blocks + 1, \
            f'channels length ({len(channels)}) should equal to num_blocks + 1 ({num_blocks + 1})'
        if skip_channels is not None:
            assert len(skip_channels) == num_blocks, \
                f'skip_channels length ({len(skip_channels)}) should equal to num_blocks ({num_blocks})'

        # Build decoder blocks
        self.blocks = ModuleList()
        for idx in range(num_blocks):
            curr_skip_channels = skip_channels[idx] if skip_channels is not None else skip_channels
            self.blocks.append(
                BasicBlock(
                    in_channels=channels[idx],
                    out_channels=channels[idx + 1],
                    inter_channels=channels[idx] // 2,
                    skip_channels=curr_skip_channels,
                    act_cfgs=act_cfgs,
                    norm_cfgs=norm_cfgs
                )
            )

        # Final disparity prediction layer
        self.pred = ConvModule(
            in_channels=channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            act_cfg=act_cfgs.get('pred'),
            norm_cfg=norm_cfgs.get('pred')
        )

    def forward(self, x: torch.Tensor, skips: Optional[List[torch.Tensor]] = None,
                pred_skips: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        """Forward function for multiscale disparity prediction.

        Args:
            x (torch.Tensor): Input feature tensor
            skips (List[torch.Tensor], optional): List of skip connection tensors from upstream encoder layers.
                Should have the same length as number of blocks. Defaults to None.
            pred_skips (List[torch.Tensor], optional): List of skip connection prediction
                                                       from upstream estimator(e.g. dispnetC).

        Returns:
            List[torch.Tensor]: List of predicted disparity maps at different scales,
                from coarse to fine resolution.
        """
        # Validate skip connections
        if skips is not None:
            assert len(skips) == len(self.blocks), \
                f'Number of skip connections ({len(skips)}) should equal to ' \
                f'number of blocks ({len(self.blocks)})'
        if pred_skips is not None:
            assert len(pred_skips) == len(self.blocks) + 1

        # Multi-scale disparity prediction
        results = []
        for idx, block in enumerate(self.blocks):
            skip = skips[idx] if skips is not None else skips
            x, pred = block(x, skip)
            pred = pred + pred_skips[idx] if pred_skips is not None else pred
            results.append(pred)

        # Final disparity prediction at the finest scale
        pred = self.pred(x) + pred_skips[-1] if pred_skips is not None else self.pred(x)
        results.append(pred)

        return results

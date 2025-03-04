from typing import Optional, Dict, Sequence, Union, Tuple
import torch.nn as nn

from mmdepth.registry import MODELS
from mmcv.cnn import ConvModule, build_activation_layer


def _make_layer_config(
        value: Union[Sequence, Dict, int, None],
        default_value: Union[Dict, int],
        num_layers: int = 2) -> Tuple:
    """Convert input config to tuple format.

    Args:
        value: Input value to be converted
        default_value: Default value if input is None
        num_layers: Number of layers in the block

    Returns:
        Tuple of configs with length num_layers
    """
    if isinstance(value, Sequence):
        assert len(value) == num_layers, f'Expected {num_layers} values, got {len(value)}'
        return tuple(value)
    elif isinstance(value, (Dict, int)) or value is None:
        value = default_value if value is None else value
        return tuple([value] * num_layers)
    else:
        raise TypeError(f'Unsupported type {type(value)}')


@MODELS.register_module()
class BasicBlock(nn.Module):
    """BasicBlock with configurable components.
    The second conv layer uses fixed kernel_size=3, stride=1, and padding=1.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (Optional[int]): Kernel size for first conv layer. Default: 3
        stride (Optional[int]): Stride for first conv layer. Default: 1
        padding (Optional[int]): Padding for first conv layer. Default: 1
        conv_cfgs (Union[Sequence[Dict], Dict, None]): Config for convolution layers.
            Default: dict(type='Conv2d')
        norm_cfgs (Union[Sequence[Dict], Dict, None]): Config for normalization layers.
            Default: dict(type='BN')
        act_cfgs (Union[Sequence[Dict], Dict, None]): Config for activation layers.
            Default: dict(type='ReLU')
        downsample: Downsample layer config or module.
            Default: None
    """

    expansion = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Optional[int] = 3,
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 1,
                 conv_cfgs: Optional[Union[Sequence[Dict], Dict]] = dict(type='Conv2d'),
                 norm_cfgs: Optional[Union[Sequence[Dict], Dict]] = dict(type='BN'),
                 act_cfgs: Optional[Union[Sequence[Dict], Dict]] = dict(type='ReLU'),
                 downsample: Optional[Union[Dict, nn.Module]] = None) -> None:
        super().__init__()

        # Convert layer-specific configs to tuples
        conv_cfgs = _make_layer_config(conv_cfgs, dict(type='Conv2d'))
        norm_cfgs = _make_layer_config(norm_cfgs, dict(type='BN'))
        act_cfgs = _make_layer_config(act_cfgs, dict(type='ReLU'))

        # Calculate intermediate channels
        mid_channels = in_channels * self.expansion

        # First conv block with configurable parameters
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfgs[0],
            norm_cfg=norm_cfgs[0],
            act_cfg=act_cfgs[0])

        # Second conv block with fixed parameters
        self.conv2 = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,  # Fixed kernel size
            stride=1,  # Fixed stride
            padding=1,  # Fixed padding
            conv_cfg=conv_cfgs[1],
            norm_cfg=norm_cfgs[1],
            act_cfg=None)  # No activation in second conv

        # Downsample handling
        self.downsample = None
        needs_downsample = (
                in_channels != out_channels or  # Channel number changes
                stride > 1  # Spatial size changes
        )

        if needs_downsample and downsample is None:
            # Auto create downsample using ConvModule if needed and not provided
            self.downsample = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                conv_cfg=conv_cfgs[0],
                norm_cfg=norm_cfgs[0],
                act_cfg=None)  # No activation in downsample
        elif isinstance(downsample, Dict):
            # Build downsample from config
            self.downsample = MODELS.build(downsample)
        elif isinstance(downsample, nn.Module):
            # Use provided downsample module
            self.downsample = downsample

        # Final activation
        self.activate = None
        if act_cfgs[1] is not None:
            self.activate = build_activation_layer(act_cfgs[1])

    def forward(self, x):
        identity = x

        # First conv block
        out = self.conv1(x)

        # Second conv block (with fixed parameters)
        out = self.conv2(out)

        # Residual connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        # Optional final activation
        if self.activate is not None:
            out = self.activate(out)

        return out


@MODELS.register_module()
class Bottleneck(nn.Module):
    """Bottleneck block with configurable components.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size for middle conv layer. Default: 3
        stride (int): Stride for middle conv layer. Default: 1
        conv_cfg (Union[Sequence[Dict], Dict, None]): Config for conv layers.
            Default: dict(type='Conv2d')
        norm_cfg (Union[Sequence[Dict], Dict, None]): Config for norm layers.
            Default: dict(type='BN')
        act_cfg (Union[Sequence[Dict], Dict, None]): Config for activation layers.
            Default: dict(type='ReLU')
    """
    #
    expansion = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 conv_cfg: Optional[Union[Sequence[Dict], Dict]] = dict(type='Conv2d'),
                 norm_cfg: Optional[Union[Sequence[Dict], Dict]] = dict(type='BN'),
                 act_cfg: Optional[Union[Sequence[Dict], Dict]] = dict(type='ReLU')) -> None:
        super().__init__()

        # Convert layer-specific configs to tuples
        conv_cfg = _make_layer_config(conv_cfg, dict(type='Conv2d'), 3)
        norm_cfg = _make_layer_config(norm_cfg, dict(type='BN'), 3)
        act_cfg = _make_layer_config(act_cfg, dict(type='ReLU'), 3)

        # First 1x1 conv for dimension reduction
        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  #
            stride=1,  #
            padding=0,  #
            conv_cfg=conv_cfg[0],
            norm_cfg=norm_cfg[0],
            act_cfg=act_cfg[0])

        # Middle 3x3 conv
        self.conv2 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,  #
            stride=stride,  #
            padding=padding,  #
            conv_cfg=conv_cfg[1],
            norm_cfg=norm_cfg[1],
            act_cfg=act_cfg[1])

        # Last 1x1 conv for dimension increase
        self.conv3 = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=1,  #
            stride=1,  #
            padding=0,  #
            conv_cfg=conv_cfg[2],
            norm_cfg=norm_cfg[2],
            act_cfg=None)  # no activation function

        # Downsample handling (自动配置)
        self.downsample = None
        if in_channels != out_channels * self.expansion or stride != 1:  # align the channel and spatial dimension
            self.downsample = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels * self.expansion,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                conv_cfg=conv_cfg[0],
                norm_cfg=norm_cfg[0],
                act_cfg=None)

    def forward(self, x):
        identity = x

        # Three conv blocks
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Residual connection with automatic downsampling
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

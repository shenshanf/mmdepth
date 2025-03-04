from typing import Tuple, Union, Sequence, List, Dict, Any, Optional
import math
import functools
import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList, Sequential
from mmcv.cnn import ConvModule

from mmdepth.registry import AGGREGATORS

from ..modules import Bottleneck as _Bottleneck


class Bottleneck(_Bottleneck):
    expansion = 1  # in aa_net paper, expansion is set 1


@AGGREGATORS.register_module(name="ISAgg")
class IntraScaleAgg(BaseModule):
    """Intra-Scale Aggregation Module for AANet.

    This module performs adaptive intra-scale aggregation by applying
    multiple deformable convolution blocks at each scale level. The channel
    number of each level is automatically calculated based on the scale_factor.
    """

    def __init__(self,
                 base_channels,
                 num_levels: int,
                 num_blocks: int,
                 scale_factor: int = 2,
                 norm_cfg: Dict = dict(type='BatchNorm2d'),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 block_conv_cfg: Dict = dict(type='DCNv2', deform_groups=2)
                 ):
        super().__init__()

        # Validate input parameters
        assert num_levels > 0, 'num_levels must be positive'
        assert num_blocks > 0, 'num_blocks must be positive'

        self.branches = ModuleList()
        for i in range(num_levels):
            blocks = []
            for _ in range(num_blocks):
                channels = base_channels // (scale_factor ** i)
                blocks.append(Bottleneck(channels, channels,
                                         norm_cfg=norm_cfg,
                                         act_cfg=act_cfg,
                                         conv_cfg=block_conv_cfg))
            self.branches.append(Sequential(*blocks))

    def forward(self, cost_vols: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function for IntraScaleAgg.

        Args:
            cost_vols: A list of cost volumes from different scales.
                Each tensor should have shape [B, C, H, W] where C is the
                number of channels for that level.

        Returns:
            A list of aggregated cost volumes
        """
        if len(cost_vols) != len(self.branches):
            raise ValueError(
                f'Expected {len(self.branches)} cost volumes, got {len(cost_vols)}'
            )

        # Process each scale level
        output_vols = []
        for cost_vol, branch in zip(cost_vols, self.branches):
            output_vols.append(branch(cost_vol))

        return output_vols


class CSADownFusion(BaseModule):
    """Cross Scale Aggregation Down-sampling Fusion module.

    This module handles the down-sampling fusion path in cross scale aggregation,
    consisting of multiple stride=2 convolution layers."""

    def __init__(self, base_channels: int,
                 scale_factor: int):
        """
        Args:
            base_channels: Number of input channels
            scale_factor: Total down-sampling factor (must be >= 1)
        """
        super().__init__()
        assert scale_factor >= 1

        if scale_factor == 1:
            self.down_layers = nn.Identity()
            return

        down_layers = []
        cur_channels = base_channels

        # Check if scale_factor is a power of 2
        assert scale_factor & (scale_factor - 1) == 0, \
            f"scale_factor {scale_factor} must be a power of 2"

        # Calculate number of down-sampling steps needed
        num_downs = int(math.log2(scale_factor))

        for i in range(num_downs):
            # All layers except last one have ReLU activation
            is_last = (i == num_downs - 1)
            down_layers.append(
                ConvModule(
                    in_channels=cur_channels,
                    out_channels=base_channels * (2 ** (i + 1)),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=dict(type='BatchNorm2d'),
                    act_cfg=None if is_last else dict(type='LeakyReLU', negative_slope=0.2, inplace=True),
                    bias='auto'
                )
            )
            cur_channels = base_channels * (2 ** (i + 1))  # in_channels of next conv layer

        self.down_layers = Sequential(*down_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return self.down_layers(x)


class CSAUpFusion(BaseModule):
    """Cross Scale Aggregation Up-sampling Fusion module.

    This module handles the up-sampling fusion path in cross scale aggregation,
    consisting of a 1x1 convolution followed by bilinear upsampling."""

    def __init__(self, base_channels: int,
                 scale_factor: int):
        """
        Args:
            base_channels: Number of input channels
            scale_factor: Up-sampling factor (must be >= 1)
        """
        super().__init__()
        assert scale_factor >= 1

        self.conv = ConvModule(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=dict(type='BatchNorm2d'),
            act_cfg=None,
            bias='auto'
        )
        self.up_layer = nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        ) if scale_factor > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return self.up_layer(self.conv(x))


@AGGREGATORS.register_module(name="CSAgg")
class CrossScaleAgg(BaseModule):
    """Cross Scale Aggregation module.

    This module performs adaptive cross-scale aggregation by fusing features
    from different scales through up/down-sampling paths.
    """

    def __init__(self, base_channels: int,
                 num_levels: int,
                 num_out_levels: int,
                 scale_factor: int = 2):
        """
        Args:
            base_channels: Base number of channels (usually max_disp)
            num_levels: Number of input feature levels
            num_out_levels: Number of output feature levels
            scale_factor: Scaling factor between adjacent levels (default: 2)
        """
        super().__init__()
        assert num_out_levels <= num_levels
        self.num_levels = num_levels
        self.num_out_levels = num_out_levels
        self.scale_factor = scale_factor

        # Create fusion layers for each output level
        self.fuse_layers = ModuleList()
        for i in range(num_out_levels):
            per_scale_layers = ModuleList()
            for j in range(num_levels):
                if j == i:
                    # Same level fusion - identity mapping
                    fuse_layer = nn.Identity()
                elif j < i:
                    # Down-sample fusion
                    fuse_layer = CSADownFusion(
                        base_channels=base_channels // (scale_factor ** j),
                        scale_factor=scale_factor ** (i - j)
                    )
                else:
                    # Up-sample fusion
                    fuse_layer = CSAUpFusion(
                        base_channels=base_channels // (scale_factor ** j),
                        scale_factor=scale_factor ** (j - i)
                    )
                per_scale_layers.append(fuse_layer)
            self.fuse_layers.append(per_scale_layers)

        self.final_layer = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, cost_vols: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        """
        Forward function.

        Args:
            cost_vols: Sequence of cost volumes from different scales

        Returns:
            Sequence of aggregated cost volumes
        """
        if len(cost_vols) != self.num_levels:
            raise ValueError(f"Expected {self.num_levels} cost volumes, got {len(cost_vols)}")

        if self.num_levels == 1:  # No fusion needed
            return cost_vols

        ag_cost_vols = []
        for fuse_layer in self.fuse_layers:
            # Process and accumulate features from each input level
            fused_features = []
            assert isinstance(fuse_layer, ModuleList)
            for fusion_op, cost_vol in zip(fuse_layer, cost_vols):
                fused = fusion_op(cost_vol)
                fused_features.append(fused)

            # Sum all fused features and apply final activation
            # use 'functools.reduce' to sum
            ag_cost_vol = self.final_layer(functools.reduce(torch.add, fused_features))
            ag_cost_vols.append(ag_cost_vol)

        return ag_cost_vols


@AGGREGATORS.register_module(name="AaptAggM")
class AdaptAggModule(BaseModule):
    """Adaptive Aggregation Module (AAModule).

    This module combines intra-scale and cross-scale aggregation to form
    a complete adaptive aggregation module. It first applies intra-scale
    aggregation to each scale independently, then performs cross-scale
    feature fusion.

    Args:
        base_channels (int): Base number of channels for the finest scale
        num_scales (int): Number of input scales to process
        num_output_scales (int): Number of output scales to produce
        num_blocks_isa (int): Number of deformable conv blocks per scale
        scale_factor (int): Scaling factor between adjacent scales
        isa_conv_cfg (Dict): Config for intra-scale conv type
        norm_cfg (Dict): Config for normalization layers
        act_cfg (Dict): Config for activation layers
    """

    def __init__(self,
                 base_channels: int,
                 num_scales: int,
                 num_output_scales: int,
                 num_blocks_isa: int = 1,
                 scale_factor: int = 2,
                 isa_conv_cfg: Dict = dict(type='DCNv2', deform_groups=2),
                 norm_cfg: Dict = dict(type='BatchNorm2d'),
                 act_cfg: Dict = dict(type='ReLU', inplace=True)):
        super().__init__()

        # Validate input parameters
        assert num_output_scales <= num_scales, \
            f'num_output_scales ({num_output_scales}) must be <= num_scales ({num_scales})'

        # Create intra-scale aggregation module for feature enhancement
        self.intra_scale_agg = IntraScaleAgg(
            base_channels=base_channels,
            num_levels=num_scales,
            num_blocks=num_blocks_isa,
            scale_factor=scale_factor,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            block_conv_cfg=isa_conv_cfg
        )

        # Create cross-scale aggregation module for multi-scale fusion
        self.cross_scale_agg = CrossScaleAgg(
            base_channels=base_channels,
            num_levels=num_scales,
            num_out_levels=num_output_scales,
            scale_factor=scale_factor
        )

    def forward(self, cost_vols: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function for AAModule.

        Args:
            cost_vols: A sequence of cost volumes from different scales
                Each tensor should be of shape [B, C, H, W] where C is the
                number of channels for that scale (base_channels/scale_factor^level)

        Returns:
            List of aggregated cost volumes, from finest to coarsest scale
        """
        # Enhance features at each scale independently
        intra_cost_vols = self.intra_scale_agg(cost_vols)

        # Fuse features across scales
        cross_cost_vols = self.cross_scale_agg(intra_cost_vols)

        return cross_cost_vols


@AGGREGATORS.register_module(name="StackAaptAggM")
class StackAdaptAggModule(BaseModule):
    """Stacked Adaptive Aggregation Module.

    Stacks multiple AdaptAggModules with configurable parameters for each module.
    Uses deformable convolutions in the last few modules specified by num_deform_stacks.

    Args:
        base_channels (int): Base number of channels (usually max_disp)
        num_stacks (int): Number of AdaptAggModule to stack
        num_scales (int): Number of input feature scales
        num_output_scales (int): Number of output scales (1 for eval, num_scales for training)
        scale_factor (int): Scaling factor between adjacent scales
        num_deform_stacks (int): Number of stacks using deformable conv
        num_blocks_isa (int): Number of blocks in each intra-scale aggregation
        deform_groups (int, optional): Number of groups in deformable conv. Default: 2
        norm_cfg (Dict): Config for normalization layers
        act_cfg (Dict): Config for activation layers
    """

    def __init__(self,
                 base_channels: int,
                 num_stacks: int,
                 num_scales: int,
                 num_output_scales: int,
                 scale_factor: int,
                 num_deform_stacks: int,
                 num_blocks_isa: int,
                 deform_groups: int = 2,
                 norm_cfg: Dict = dict(type='BatchNorm2d'),
                 act_cfg: Dict = dict(type='ReLU', inplace=True)):
        super().__init__()
        assert num_output_scales <= num_scales
        assert num_deform_stacks <= num_stacks

        # Build stacked modules
        stacked_modules = []
        for idx in range(num_stacks):
            # For intermediate modules, output all scales
            # For final module, output specified number of scales
            curr_output_scales = num_scales if idx < num_stacks - 1 else num_output_scales

            # Use deformable conv for last num_deform_stacks modules
            conv_cfg = dict(type='DCNv2', deform_groups=deform_groups) \
                if idx >= num_stacks - num_deform_stacks else dict(type='Conv2d')

            stacked_modules.append(
                AdaptAggModule(
                    base_channels=base_channels,
                    num_scales=num_scales,
                    num_output_scales=curr_output_scales,
                    num_blocks_isa=num_blocks_isa,
                    scale_factor=scale_factor,
                    isa_conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )

        self.stacked_modules = Sequential(*stacked_modules)

        # Create final 1x1 convolution layers for each output scale
        final_layers = ModuleList()
        for i in range(num_output_scales):
            channels = base_channels // (scale_factor ** i)
            final_layers.append(nn.Conv2d(channels, channels, 1))

        self.final_layers = final_layers

    def forward(self, cost_volumes: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Forward function for stacked adaptive aggregation modules.

        Args:
            cost_volumes: Multi-scale cost volumes to process

        Returns:
            Processed multi-scale cost volumes after stacked aggregation
        """
        # Pass through stacked adaptive aggregation modules
        processed_volumes = self.stacked_modules(cost_volumes)

        # Apply final 1x1 convolutions
        output_volumes = [
            final_layer(volume)
            for volume, final_layer in zip(processed_volumes, self.final_layers)
        ]

        return output_volumes

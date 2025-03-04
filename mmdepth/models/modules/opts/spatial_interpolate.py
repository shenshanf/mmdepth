import torch
import torch.nn.functional as F
from typing import Literal, Union


class SpatialInterpol:
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        """
        Initialize interpolation parameters

        Args:
            size: int or tuple - Target output size
            scale_factor: float or tuple - Multiplier for spatial size
            mode: str - Algorithm used for interpolation ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', etc.)
            align_corners: bool - If True, the corner pixels of input and output tensors are aligned (only for some modes)
        """
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, x):
        """
        Perform interpolation operation

        Args:
            x: torch.Tensor - Input tensor to be interpolated

        Returns:
            torch.Tensor - Interpolated tensor with desired size
        """
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners if self.mode != 'nearest' else None
        )


class Spatial2DInterpolAs:
    def __init__(self, mode, align_corners):
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, x, target):
        return spatial_2d_interpol_as(x, target, self.mode, self.align_corners)


class Spatial3DInterpolAs:
    def __init__(self, mode, align_corners):
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, x, target):
        return spatial_3d_interpol_as(x, target, self.mode, self.align_corners)


def spatial_2d_interpol_as(
        x: torch.Tensor,
        target: torch.Tensor,
        mode: Literal['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'] = 'bilinear',
        align_corners: bool = False
) -> torch.Tensor:
    """Interpolate input tensor x to match spatial dimensions of target tensor.

    Args:
        x: Input tensor with shape [batch_size, channels, height_1, width_1]
        target: Target tensor with shape [batch_size, channels, height_2, width_2]
        mode: Interpolation mode, one of ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
        align_corners: If True, align the corners of input and output. Only valid when mode is not 'nearest'

    Returns:
        torch.Tensor: Interpolated tensor with same spatial dimensions as target

    Raises:
        ValueError: When tensor dimensions are incorrect or mode is invalid
    """
    # Check input dimensions
    if x.dim() != 4 or target.dim() != 4:
        raise ValueError("Both input x and target must be 4D tensors [batch, channels, height, width]")

    # Check if batch and channel dimensions match
    if x.shape[:2] != target.shape[:2]:
        raise ValueError("Batch and channel dimensions must match between input x and target")

    # If spatial dimensions already match, return input directly
    if x.shape[-2:] == target.shape[-2:]:
        return x

    # Validate interpolation mode
    valid_modes = {'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'}
    if mode not in valid_modes:
        raise ValueError(f"Interpolation mode must be one of: {valid_modes}")

    # Perform interpolation
    return F.interpolate(
        x,
        size=target.shape[-2:],  # Only take height and width
        mode=mode,
        align_corners=align_corners if mode != 'nearest' else None
    )


def spatial_3d_interpol_as(
        x: torch.Tensor,
        target: torch.Tensor,
        mode: Literal['nearest', 'linear', 'trilinear'] = 'trilinear',
        align_corners: bool = False
) -> torch.Tensor:
    """Interpolate input tensor x to match spatial dimensions of target tensor in 3D space.

    Args:
        x: Input tensor with shape [batch_size, channels, depth_1, height_1, width_1]
        target: Target tensor with shape [batch_size, channels, depth_2, height_2, width_2]
        mode: Interpolation mode, one of ['nearest', 'linear', 'trilinear']
        align_corners: If True, align the corners of input and output. Only valid when mode is not 'nearest'

    Returns:
        torch.Tensor: Interpolated tensor with same spatial dimensions as target

    Raises:
        ValueError: When tensor dimensions are incorrect or mode is invalid
    """
    # Check input dimensions
    if x.dim() != 5 or target.dim() != 5:
        raise ValueError("Both input x and target must be 5D tensors [batch, channels, depth, height, width]")

    # Check if batch and channel dimensions match
    if x.shape[:2] != target.shape[:2]:
        raise ValueError("Batch and channel dimensions must match between input x and target")

    # If spatial dimensions already match, return input directly
    if x.shape[-3:] == target.shape[-3:]:
        return x

    # Validate interpolation mode
    valid_modes = {'nearest', 'linear', 'trilinear'}
    if mode not in valid_modes:
        raise ValueError(f"Interpolation mode must be one of: {valid_modes}")

    # Perform interpolation
    return F.interpolate(
        x,
        size=target.shape[-3:],  # Take depth, height and width
        mode=mode,
        align_corners=align_corners if mode != 'nearest' else None
    )

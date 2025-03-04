import torch
import torch.nn.functional as F


def spatial_pad_as(x: torch.Tensor,
                   target: torch.Tensor,
                   mode: str = 'constant',
                   value: float = 0.0) -> torch.Tensor:
    """Pad the spatial dimensions of source tensor to match target tensor.

    This function handles N-dimensional spatial alignment through padding.
    The padding is applied symmetrically from both sides of each spatial dimension.

    Args:
        x (torch.Tensor): Source tensor to be padded, shape (B, C, *spatial_dims)
        target (torch.Tensor): Target tensor that provides desired spatial dimensions
        mode (str, optional): Padding mode, supported modes: 'constant', 'reflect',
                            'replicate', 'circular'. Defaults to 'constant'
        value (float, optional): Fill value for constant padding. Defaults to 0.0

    Returns:
        torch.Tensor: Padded tensor with spatial dimensions matching target tensor

    Example:
        >>> # Align 3D feature maps
        >>> x0 = torch.randn(1, 32, 16, 32, 32)          # source feature
        >>> target0 = torch.randn(1, 16, 32, 64, 64)     # target feature
        >>> aligned_x0 = spatial_pad_as(x, target)       # output: (1, 32, 32, 64, 64)

    Raises:
        ValueError: If target spatial dimensions are smaller than source
        ValueError: If tensors have different number of spatial dimensions
    """
    # Extract spatial dimensions (excluding batch and channel dims)
    x_spatial = x.shape[2:]
    target_spatial = target.shape[2:]

    # Validate dimensions
    if len(x_spatial) != len(target_spatial):
        raise ValueError(
            f"Incompatible number of spatial dimensions: "
            f"source {len(x_spatial)}D vs target {len(target_spatial)}D"
        )

    if x_spatial == target_spatial:  # already spatial align, no need pad
        return x

    # Calculate required padding per dimension
    diff_dims = [t - s for s, t in zip(x_spatial, target_spatial)]

    # Validate target size
    if any(d < 0 for d in diff_dims):
        raise ValueError(
            f"Target spatial dims must be larger or equal to source dims. "
            f"Source: {x_spatial}, Target: {target_spatial}"
        )

    # Construct padding tuple (pad_left, pad_right, pad_top, pad_bottom, ...)
    # Note: F.pad expects dimensions in reverse order
    paddings = []
    for d in diff_dims[::-1]:
        # Distribute padding evenly on both sides
        # For odd padding, extra pixel goes to right/bottom/back
        paddings.extend([d // 2, d - d // 2])

    return F.pad(x, paddings, mode=mode, value=value)

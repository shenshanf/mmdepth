import torch.nn as nn
import torch.nn.functional as F
import torch


@torch.no_grad()
def mesh_grid2D(bs: int, ht: int, wd: int,
                dtype: torch.dtype = torch.float32,
                device: torch.device = torch.device('cuda')) -> torch.Tensor:
    """Generate 2D coordinates grid with batch dimension.

    Args:
        bs: Batch size
        ht: Height of the grid
        wd: Width of the grid
        dtype: Data type of output tensor
        device: Target device for the grid. If None, uses default device

    Returns:
        torch.Tensor: Grid coordinates of shape [B, H, W, 2], where the last dimension
                     contains [x, y] coordinates for each position
    """
    # Generate coordinate arrays
    x = torch.arange(wd, device=device, dtype=dtype)  # [W]
    y = torch.arange(ht, device=device, dtype=dtype)  # [H]

    # Create 2D meshgrid
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # [H, W]

    # Stack x,y coordinates and add batch dimension
    # Shape progression: [H, W] -> [H, W, 2] -> [1, H, W, 2] -> [B, H, W, 2]
    coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    coords = coords.unsqueeze(0).expand(bs, -1, -1, -1)  # [B, H, W, 2]

    return coords.contiguous()


@torch.no_grad()
def mesh_grid2D_x(bs: int, ht: int, wd: int,
                  dtype: torch.dtype = torch.float32,
                  device: torch.device = torch.device('cuda')) -> torch.Tensor:
    """Generate 2D coordinates grid x with batch dimension.

    Args:
        bs: Batch size
        ht: Height of the grid
        wd: Width of the grid
        dtype: Data type of output tensor
        device: Target device for the grid. If None, uses default device

    Returns:
    """
    # Generate coordinate arrays
    x = torch.arange(wd, device=device, dtype=dtype)  # [W]
    grid_x = x[None, None, None, :].expand(bs, -1, ht, -1)

    return grid_x


@torch.no_grad()
def mesh_grid2D_y(bs: int, ht: int, wd: int,
                  dtype: torch.dtype = torch.float32,
                  device: torch.device = torch.device('cuda')) -> torch.Tensor:
    """Generate 2D coordinates grid x with batch dimension.

    Args:
        bs: Batch size
        ht: Height of the grid
        wd: Width of the grid
        dtype: Data type of output tensor
        device: Target device for the grid. If None, uses default device

    Returns:
    """
    # Generate coordinate arrays
    y = torch.arange(ht, device=device, dtype=dtype)  # [H]
    grid_y = y[None, None, :, None].expand(bs, -1, -1, wd)

    return grid_y


@torch.no_grad()
def mesh_grid3D(bs: int, dt: int, ht: int, wd: int,
                dtype: torch.dtype = torch.float32,
                device: torch.device = torch.device('cuda')) -> torch.Tensor:
    """Generate 3D coordinates grid with batch dimension.

    Args:
        bs: Batch size
        dt: Depth of the grid
        ht: Height of the grid
        wd: Width of the grid
        dtype: Data type of output tensor
        device: Target device for the grid

    Returns:
        torch.Tensor: Grid coordinates of shape [B, D, H, W, 3], where the last dimension
                     contains [x, y, z] coordinates for each position
    """
    # Generate coordinate arrays
    x = torch.arange(wd, device=device, dtype=dtype)  # [W]
    y = torch.arange(ht, device=device, dtype=dtype)  # [H]
    z = torch.arange(dt, device=device, dtype=dtype)  # [D]

    # Create 3D meshgrid
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')  # [D, H, W]

    # Stack x,y,z coordinates and add batch dimension
    # Shape progression: [D, H, W] -> [D, H, W, 3] -> [1, D, H, W, 3] -> [B, D, H, W, 3]
    coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [D, H, W, 3]
    coords = coords.unsqueeze(0).expand(bs, -1, -1, -1, -1)  # [B, D, H, W, 3]

    return coords.contiguous()


@torch.no_grad()
def mesh_grid2D_like(feat_map: torch.Tensor, dtype=None) -> torch.Tensor:
    """Generate 2D coordinates grid matching the input feature map's spatial dimensions.

    Args:
        feat_map: Input tensor of shape [B, H, W] or [B, C, H, W]
        dtype
    Returns:
        torch.Tensor: Grid coordinates of shape [B, H, W, 2], where the last dimension
                     contains [x, y] coordinates for each position
    """
    if len(feat_map.shape) == 3:
        bs, ht, wd = feat_map.shape
    elif len(feat_map.shape) == 4:
        bs, _, ht, wd = feat_map.shape
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {feat_map.shape}")

    if dtype is None:
        dtype = feat_map.dtype

    return mesh_grid2D(
        bs=bs,
        ht=ht,
        wd=wd,
        dtype=dtype,
        device=feat_map.device
    )


@torch.no_grad()
def mesh_grid3D_like(feat_vol: torch.Tensor, dtype=None) -> torch.Tensor:
    """Generate 3D coordinates grid matching the input feature volume's spatial dimensions.

    Args:
        feat_vol: Input tensor of shape [B, D, H, W] or [B, C, D, H, W]
        dtype
    Returns:
        torch.Tensor: Grid coordinates of shape [B, D, H, W, 3], where the last dimension
                     contains [x, y, z] coordinates for each position
    """
    if len(feat_vol.shape) == 4:
        bs, dt, ht, wd = feat_vol.shape
    elif len(feat_vol.shape) == 5:
        bs, _, dt, ht, wd = feat_vol.shape
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got shape {feat_vol.shape}")

    if dtype is None:
        dtype = feat_vol.dtype

    return mesh_grid3D(
        bs=bs,
        dt=dt,
        ht=ht,
        wd=wd,
        dtype=dtype,
        device=feat_vol.device
    )


def norm_grid2D(coords: torch.Tensor) -> torch.Tensor:
    """Normalize 2D coordinates from pixel coordinates to [-1, 1] range.

    Args:
        coords: Input coordinates of shape [B, H, W, 2], where last dimension
               contains [x, y] pixel coordinates

    Returns:
        torch.Tensor: Normalized coordinates in [-1, 1] range with same shape as input
    """
    _, H, W, _ = coords.shape

    # Scale x coordinates from [0, W-1] to [-1, 1]
    coords[..., 0] = 2.0 * coords[..., 0] / (W - 1) - 1.0

    # Scale y coordinates from [0, H-1] to [-1, 1]
    coords[..., 1] = 2.0 * coords[..., 1] / (H - 1) - 1.0

    return coords


def norm_grid3D(coords: torch.Tensor) -> torch.Tensor:
    """Normalize 3D coordinates from pixel coordinates to [-1, 1] range.

    Args:
        coords: Input coordinates of shape [B, D, H, W, 3], where last dimension
               contains [x, y, z] pixel coordinates

    Returns:
        torch.Tensor: Normalized coordinates in [-1, 1] range with same shape as input
    """
    _, D, H, W, _ = coords.shape

    # Scale x coordinates from [0, W-1] to [-1, 1]
    coords[..., 0] = 2.0 * coords[..., 0] / (W - 1) - 1.0

    # Scale y coordinates from [0, H-1] to [-1, 1]
    coords[..., 1] = 2.0 * coords[..., 1] / (H - 1) - 1.0

    # Scale z coordinates from [0, D-1] to [-1, 1]
    coords[..., 2] = 2.0 * coords[..., 2] / (D - 1) - 1.0

    return coords


if __name__ == "__main__":
    grid = mesh_grid2D(1, 5, 4)
    pass

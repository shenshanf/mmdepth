from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdepth.models.modules.opts import (mesh_grid2D_like, mesh_grid3D_like,
                                         norm_grid2D, norm_grid3D, mesh_grid2D_x, mesh_grid2D_y)


class BaseWarpOpt(BaseModule, ABC):
    def __init__(self,
                 mode: str = 'bilinear',
                 padding_mode: str = 'border',
                 use_mask: bool = False) -> None:
        super().__init__()
        """Initialize WarpOpt module.

        Args:
            mode: Interpolation mode for grid sampling
            padding_mode: Padding mode for out-of-bounds grid locations
        """
        self.mode = mode
        self.padding_mode = padding_mode

        # if use_mask:
        #     raise NotImplementedError(f'use_mask not implemented')
        self.use_mask = use_mask

    @abstractmethod
    def forward(self, feat: torch.Tensor, sample_grid: torch.Tensor):
        ...


class WarpOpt2D(BaseWarpOpt):
    """
    """

    def forward(self, feat, sample_grid):
        """

        Args:
            feat: [b,c,h,w]
            sample_grid: [b,h,w,2] relative shift at coord_x, coord_y
                                     positive means shift left
                         [b,h,w] relative shift at coord_x

        Returns:
            warped_feat: [b,c,h,w]
        """
        if sample_grid.ndim == 3:  # [b,h,w]
            # [b,h,w,2] <- [b,h,w]
            coords = mesh_grid2D_like(sample_grid)
            # shift x
            # [b,h,w,2(idx=0)] = [b,h,w,2(idx=0)] + [b,h,w]
            coords[..., 0] = coords[..., 0] + sample_grid
        elif sample_grid.ndim == 4:  # [b,h,w,2]
            # [b,h,w,2] <- [b,h,w] <- [b,h,w,2]
            coords = mesh_grid2D_like(sample_grid[..., 0])
            # shift x and y
            # [b,h,w,2] = [b,h,w,2] + [b,h,w,2]
            coords = coords + sample_grid
        else:
            raise ValueError

        # [b,h,w,2]
        coords = norm_grid2D(coords)

        # [b,c h,w] <= [b,c,h,w] & [b,h,w,2]
        feat_warped = F.grid_sample(feat, coords,
                                    mode=self.mode,
                                    padding_mode=self.padding_mode,
                                    align_corners=True)
        # [b,c h,w]
        return feat_warped


class GatherOpt2D(BaseWarpOpt):
    """Gather operation along H or W dimension"""

    def __init__(self, dim='w'):
        """
        Args:
            dim: Dimension to perform gather operation, 'h' or 'w'
        """
        super().__init__()
        assert dim in ['h', 'w']
        self.dim = dim

    def forward(self, feat, sample_grid):
        """
        Args:
            feat: [b,c,h,w]
            sample_grid: [b,1,h,w] integer indices for offset in specified dimension
                        positive values indicate upward/leftward shifts
                        or int

        Returns:
            gathered_feat: [b,c,h,w]
        """
        b, c, h, w = feat.shape

        if self.dim == 'w':
            # Gather along w dimension
            base_grid = mesh_grid2D_x(b, h, w, dtype=feat.dtype, device=feat.device)

            # Compute sampling indices
            # [b,1,h,w]/scale -> [b,1,h,w]
            sample_indices = base_grid + sample_grid

            # Ensure valid index range
            sample_indices = torch.clamp(sample_indices, 0, w - 1)

            # Gather operation
            # [b,c,h,w] <- gather from [b,c,h,w] at dim=3 with index [b,1,h,w]
            gathered_feat = torch.gather(
                feat,
                dim=3,
                index=sample_indices.expand(b, c, h, w)
            )
        else:
            # Gather along h dimension
            base_grid = mesh_grid2D_y(b, h, w, dtype=feat.dtype, device=feat.device)

            # Compute sampling indices
            # [b,h,w] -> [b,1,h,w]
            sample_indices = base_grid + sample_grid

            # Ensure valid index range
            sample_indices = torch.clamp(sample_indices, 0, h - 1)

            # Gather operation
            # [b,c,h,w] <- gather from [b,c,h,w] at dim=2 with index [b,1,h,w]
            gathered_feat = torch.gather(
                feat,
                dim=2,
                index=sample_indices.expand(b, c, h, w)
            )

        return gathered_feat


class WarpOpt3D(BaseWarpOpt):

    def forward(self, feat, sample_grid):
        """

        Args:
            feat: [b,c,h,w]
            sample_grid: [b,d,h,w,2] relative shift at coord_x, coord_y
                                     positive means shift left
                         [b,d,h,w] relative shift at coord_x

        Returns:
            warped_feat: [b,c,d,h,w]
        """

        if sample_grid.ndim == 4:  # [b,d,h,w]
            # [b,d,h,w,3(x,y,z)]
            coords = mesh_grid3D_like(sample_grid)
            # [b,d,h,w, 3(x)] = [b,d,h,w, 3(x)] + [b,d,h,w]
            coords[..., 0] = coords[..., 0] + sample_grid
        elif sample_grid.ndim == 5:  # [b,d,h,w,2]
            # [b,d,h,w,3] <- [b,d,h,w] <- [b,d,h,w,2]
            coords = mesh_grid3D_like(sample_grid[..., 0])
            # [b,h,w,3(x,y)] <- [b,d,h,w,3(x,y)] + [b,d,h,w,2(x,y)]
            coords[..., 0] = coords[..., 0] + sample_grid[..., 0]
            coords[..., 1] = coords[..., 1] + sample_grid[..., 1]
        else:
            raise ValueError

        # [b,d,h,w,3]
        coords = norm_grid3D(coords)

        # [b,c,h,w] -> [b,c,d=1,h,w] -> [b,c,d,h,w]
        feat = feat[:, :, None, :, :].expand(-1, -1, sample_grid.shape[1], -1, -1)

        # [b,c,d,h,w] <= [b,c,d,h,w] & [b,d,h,w,3]
        feat_warped = F.grid_sample(feat, coords,
                                    mode=self.mode,
                                    padding_mode=self.padding_mode,
                                    align_corners=True)
        # [b,c,d,h,w]
        return feat_warped


class LookUpOpt(BaseWarpOpt):

    def __init__(self, mode: str = 'bilinear',
                 padding_mode: str = 'border',
                 use_mask: bool = False,
                 volume_format='DHW'):

        super().__init__(mode, padding_mode,
                         use_mask)
        assert volume_format in ['DHW', 'HWW']
        self.volume_format = volume_format

    def _dhw_forward(self, volume: torch.Tensor, sample_grid: torch.Tensor):
        """

        Args:
            volume: [b,c,d0,h,w], global cost volume
                    note：你必须确定d0维度上的视差和维度坐标一致
            sample_grid: [b,d1,h,w],

        Returns:
            local volume: [b,c,d1,h,w]
        """
        if volume.ndim == 4:
            # [b,d0,h,w] -> [b,c=1,d0,h,w]
            volume = volume.unsqueeze(dim=1)
        #
        coords = mesh_grid3D_like(sample_grid)

        #
        coords[..., 2] = sample_grid

        #
        coords = norm_grid3D(coords)
        #
        local_volume = F.grid_sample(volume, coords,
                                     mode=self.mode,
                                     padding_mode=self.padding_mode,
                                     align_corners=True)

        return local_volume

    def _hww_forward(self, volume: torch.Tensor, sample_grid: torch.Tensor):
        """

        Args:
            volume: [b,c,h,w,w1]
            sample_grid: [b,d1,h,w] lookup at 'w1' dimension

        Returns:
            local volume: [b,d1,h,w]
        """
        if volume.ndim == 4:
            # [b,h,w,w1] -> [b,c=1,h,w,w1]
            volume = volume.unsqueeze(dim=1)

        # [b,d1,h,w,3] <- [b,d1,h,w]
        coords = mesh_grid3D_like(sample_grid)

        # [b,w1,h,w,3(x1)] = [b,d1,h,w,3(z)] + [b,d1,h,w,3(x)]
        coords[..., 2] = coords[..., 2] + coords[..., 1]
        # [b,w1,h,w,3] -> [b,h,w,w1,3]
        coords = coords.permute((0, 2, 3, 1, 4))
        #
        coords = norm_grid3D(coords)
        #
        local_volume = F.grid_sample(volume, coords,
                                     mode=self.mode,
                                     padding_mode=self.padding_mode,
                                     align_corners=True)
        return local_volume

    def forward(self, volume: torch.Tensor, sample_grid: torch.Tensor):
        if self.volume_format == 'DHW':
            return self._dhw_forward(volume, sample_grid)
        elif self.volume_format == 'HWW':
            return self._hww_forward(volume, sample_grid)
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    ...

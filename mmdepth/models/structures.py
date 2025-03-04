from typing import Optional
from dataclasses import dataclass
import torch


@dataclass
class CostVolume:
    data: torch.Tensor  # [b,d,h,w] or [b,c,d,h,w]
    # disparity sample candidates at 'd' dimension
    sample_grid: torch.Tensor  # [b,d,h,w] or [b=1,d,h=1,w=1]

    def __post_init__(self):
        if self.data is None:
            raise ValueError("data tensor must be provided")

        # Shape check
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")

        if len(self.data.shape) not in [4, 5]:
            raise ValueError("data must have shape [b,d,h,w] or [b,c,d,h,w]")

        if self.sample_grid is not None:
            if not isinstance(self.sample_grid, torch.Tensor):
                raise TypeError("sample_grid must be a torch.Tensor")

            if len(self.sample_grid.shape) != 4:
                raise ValueError("sample_grid must have shape [b,d,h,w] or [b=1,d,h=1,w=1]")

            # Check batch size compatibility
            if self.sample_grid.shape[0] != 1 and self.sample_grid.shape[0] != self.data.shape[0]:
                raise ValueError("sample_grid batch size must be 1 or match data batch size")

            # Check depth dimension compatibility
            if self.sample_grid.shape[1] != self.data.shape[-3]:
                raise ValueError("sample_grid depth dimension must match data depth dimension")

            # Check if it's a single spatial point or matches spatial dimensions
            spatial_valid = (
                    (self.sample_grid.shape[2:] == torch.Size([1, 1])) or
                    (self.sample_grid.shape[2:] == self.data.shape[-2:])
            )
            if not spatial_valid:
                raise ValueError("sample_grid spatial dimensions must be [1,1] or match data spatial dimensions")
        else:
            # create optional disp_hyps
            # the index of 'd' dimension equal to disparity sample candidates
            dp = self.data.shape[-3]  # [b,d,h,w] or [b,c,d,h,w]
            # [d,] -> [b=1,d,h=1,w=1]
            hyps = torch.arange(dp, dtype=self.data.dtype, device=self.data.device)
            self.sample_grid = hyps[None, :, None, None]


@dataclass
class DispMap:
    data: torch.Tensor = None  # [b,1,h,w]
    v_mask: Optional[torch.Tensor] = None  # [b,1,h,w]

    @property
    def is_sparse(self):
        return self.v_mask is None or torch.all(self.v_mask)

    def __post_init__(self):
        # Check if data is provided
        if self.data is None:
            raise ValueError("data tensor must be provided")

        # Check data type
        if not isinstance(self.data, torch.Tensor):
            raise TypeError("data must be a torch.Tensor")

        # Check data dimensions
        if len(self.data.shape) != 4:
            raise ValueError(f"data must have 4 dimensions [b,1,h,w], got shape {self.data.shape}")

        if self.data.shape[1] != 1:
            raise ValueError(f"data must have 1 channel, got {self.data.shape[1]} channels")

        # Check v_mask if provided
        if self.v_mask is not None:
            # Check mask type
            if not isinstance(self.v_mask, torch.Tensor):
                raise TypeError("v_mask must be a torch.Tensor")

            # Check mask dimensions
            if len(self.v_mask.shape) != 4:
                raise ValueError(f"v_mask must have 4 dimensions [b,1,h,w], got shape {self.v_mask.shape}")

            if self.v_mask.shape[1] != 1:
                raise ValueError(f"v_mask must have 1 channel, got {self.v_mask.shape[1]} channels")

            # Check compatibility between data and mask
            if self.v_mask.shape != self.data.shape:
                raise ValueError(f"v_mask shape {self.v_mask.shape} must match data shape {self.data.shape}")

            # Check if mask contains valid boolean or float values
            if self.v_mask.dtype != torch.bool:
                raise ValueError("v_mask must be boolean tensor")

            # if self.v_mask.dtype in [torch.float16, torch.float32, torch.float64]:
            #     if not ((self.v_mask >= 0) & (self.v_mask <= 1)).all():
            #         raise ValueError("Float v_mask values must be in range [0, 1]")

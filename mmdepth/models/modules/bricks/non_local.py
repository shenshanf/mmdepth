import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
from einops import rearrange
from mmcv.cnn import NonLocal3d as MMNonLocal3d


class NonLocal3d(MMNonLocal3d):
    """Extended Non-local 3D Block with multi-head attention and window partitioning.

    This class extends the MMNonLocal3d to support:
    1. Multi-head attention by splitting channels
    2. Window partitioning in D, H, W dimensions
    3. reflect padding for non-divisible spatial dimensions

    Args:
        in_channels (int): Number of input channels
        sub_sample (bool): Whether to apply max pooling after pairwise function. Default: False
        num_heads (int): Number of attention heads. Default: 8
        window_size: Size of windows for D, H, W dimensions. Default: (4, 4, 4)
        conv_cfg (Dict): Config dict for convolution layers. Default: dict(type='Conv3d')
    """

    def __init__(self, in_channels: int,
                 sub_sample: bool = False,
                 num_heads: int = 8,
                 window_size: Optional[Union[Tuple[int, int, int], int]]= 4,
                 conv_cfg: Dict = dict(type='Conv3d'),
                 **kwargs):
        """in_channels: int,
                 reduction: int = 2,
                 use_scale: bool = True,
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 mode: str = 'embedded_gaussian',"""
        # Ensure channel dimension is divisible by number of heads
        assert in_channels % num_heads == 0

        # Initialize parent class with channels divided by number of heads
        super().__init__(in_channels // num_heads, sub_sample, conv_cfg, **kwargs)

        # Store configuration
        self.num_heads = num_heads

        # Handle window_size input
        if window_size is None or window_size == -1:
            self.window_size = None
        elif isinstance(window_size, int):
            self.window_size = (window_size, window_size, window_size)
        else:
            self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for NonLocal3d with multi-head attention and window partitioning.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, depth, height, width]

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # Get input dimensions
        B, C, D, H, W = x.shape

        # 1. No window partitioning, only multi-head
        if self.window_size is None:
            x = rearrange(x, 'b (nh c) d h w -> (b nh) c (d h w)',
                          nh=self.num_heads,
                          c=C // self.num_heads)

            # Apply non-local operation from parent class
            x = super().forward(x)

            # Reshape back to original format
            x = rearrange(x, '(b nh) c (d h w) -> b (nh c) d h w',
                          b=B, nh=self.num_heads,
                          d=D, h=H, w=W)

            return x.contiguous()
        # 2. window partitioning and  multi-head

        win_d, win_h, win_w = self.window_size

        # Calculate required padding for each dimension
        pad_d = (win_d - D % win_d) % win_d
        pad_h = (win_h - H % win_h) % win_h
        pad_w = (win_w - W % win_w) % win_w

        # Apply circular padding if needed
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # note: We use reflect padding and
            #       do not mask the attention influence of padded points
            x = F.pad(x, pad=(0, pad_w, 0, pad_h, 0, pad_d), mode='reflect')

        # Get padded dimensions
        _, _, D_pad, H_pad, W_pad = x.shape

        # Calculate number of windows for each dimension
        num_win_d = D_pad // win_d
        num_win_h = H_pad // win_h
        num_win_w = W_pad // win_w

        # Reshape tensor for multi-head attention and window partitioning
        # Split channels for multi-head, and reshape spatial dimensions into windows
        x = rearrange(x, 'b (nh c) (nwd wd) (nwh wh) (nww ww) -> (b nh nwd nwh nww) c (wd wh ww)',
                      nh=self.num_heads,
                      c=C // self.num_heads,
                      nwd=num_win_d, wd=win_d,
                      nwh=num_win_h, wh=win_h,
                      nww=num_win_w, ww=win_w)

        # Apply non-local operation from parent class
        x = super().forward(x)

        # Reshape back to original format
        x = rearrange(x, '(b nh nwd nwh nww) c (wd wh ww) -> b (nh c) (nwd wd) (nwh wh) (nww ww)',
                      b=B, nh=self.num_heads,
                      nwd=num_win_d, wd=win_d,
                      nwh=num_win_h, wh=win_h,
                      nww=num_win_w, ww=win_w)

        # Remove padding if necessary
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :, :D, :H, :W]

        return x.contiguous()

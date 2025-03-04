from typing import Optional, Callable
from abc import ABC, abstractmethod, abstractproperty
import torch
import torch.nn as nn

from mmdepth.models.modules.opts import spatial_pad_as


class BaseUnetDecoder(nn.Module, ABC):
    def __init__(self, first_block: Optional[nn.Module],
                 fusion: Optional[Callable],
                 skip_block: Optional[nn.Module],
                 final_block: Optional[nn.Module]):
        super().__init__()
        self.first_block = first_block
        self.fusion = fusion
        self.skip_block = skip_block
        self.final_block = final_block

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor]):
        # first block before fusion
        x = self.first_block(x) if self.first_block is not None else x

        if skip is not None and self.fusion is not None:
            # process skip
            skip = self.skip_block(skip) if self.skip_block is not None else skip
            # fusion
            x = self.fusion(x, skip)

        # final block after fusion
        return self.final_block(x) if self.final_block is not None else x

    @staticmethod
    def cat_fusion(x, skip):
        x = spatial_pad_as(x, skip, mode='replicate')
        return torch.cat((x, skip), dim=1)

    @staticmethod
    def add_fusion(x, skip):
        x = spatial_pad_as(x, skip, mode='replicate')
        assert x.shape == skip.shape
        return torch.add(x, skip)


class BaseUNet(nn.Module, ABC):
    def __init__(self,
                 encoders: nn.ModuleList,
                 decoders: nn.ModuleList,
                 neck_layer: Optional[nn.Module] = None,
                 skip_first: bool = False,
                 multi_out: bool = False):
        super().__init__()
        # check lens
        if skip_first:
            assert len(encoders) == len(decoders), \
                f"num encoders:{len(encoders)}|num decoders:{len(decoders)}"
        else:
            assert len(encoders) == len(decoders) - 1, \
                f"num encoders:{len(encoders)}|num decoders:{len(decoders)}"

        #
        self.encoders = encoders
        self.decoders = decoders
        self.neck_layer = neck_layer
        #
        self.skip_first = skip_first
        self.multi_out = multi_out

    def forward(self, x: torch.Tensor):
        """
        Args:
            x:

        Returns:

        """
        results = []  # output results
        skips = []  # skip connection residual

        # skip connection the first input data
        if self.skip_first:
            skips.append(x)

        # encode stage
        for encode in self.encoders:
            x = encode(x)
            skips.append(x)  # append the output of encoder as skip connection

        # bottleneck stage
        x = skips.pop()
        x = self.neck_layer(x) if self.neck_layer is not None else x  # innermost data is not skip connection
        results.append(x)

        for idx in reversed(range(len(self.decoders))):  # decode and skip connection the encoder stage
            x = self.decoders[idx](x, skips.pop())
            results.append(x)

        # output style choose
        if self.multi_out:
            return results
        else:
            return results[-1]

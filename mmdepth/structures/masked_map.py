from typing import Optional, Union, Tuple, Any
import torch
import numpy as np
import mmcv

from .base import BaseDataElement, numpy_only
from mmdepth.utils import resize_sparse_map, imcrop, im_unpad, parse_padding, intersect_masks


class MaskedMap(BaseDataElement):
    """Base class for 2D map data with mask support.

    This class manages a 2D data map and its corresponding valid mask.
    Both the map and valid mask must have the same spatial dimensions (H, W).

    Attributes:
        _map: Main 2D data array/tensor
        _v_mask: Boolean mask indicating valid regions
    """

    def __init__(self, *, metainfo=None, **kwargs) -> None:
        self._map = None
        self._v_mask = None
        super().__init__(metainfo=metainfo, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override attribute setting to restrict settable attributes."""
        if name in ('_metainfo_fields', '_data_fields', 'map', 'v_mask'):
            super().__setattr__(name, value)
        elif name in ('_map', '_v_mask'):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Can only set attributes 'map' or 'v_mask', but got '{name}'")

    @property
    def map(self) -> Union[torch.Tensor, np.ndarray, None]:
        """Get the data map."""
        return self._map

    @map.setter
    def map(self, value: Union[torch.Tensor, np.ndarray]) -> None:
        """Set the data map.

        Args:
            value: Data map with shape (H, W)

        Raises:
            TypeError: If value type is incorrect
            ValueError: If dimensions or shape are incorrect
        """
        if value is None:
            self._map = None
            return

        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise TypeError(f'map must be torch.Tensor or np.ndarray, got {type(value)}')

        self._validate_shape(value)

        if self._v_mask is not None and value.shape != self._v_mask.shape:
            raise ValueError(
                f'Shape mismatch: map {value.shape} != valid_mask {self._v_mask.shape}')

        self.set_field(value, '_map', field_type='data')

    @property
    def v_mask(self) -> Union[torch.Tensor, np.ndarray, None]:
        """Get the valid mask."""
        return self._v_mask

    @v_mask.setter
    def v_mask(self, value: Optional[Union[torch.Tensor, np.ndarray]]) -> None:
        """Set the valid mask.

        Args:
            value: Valid mask with shape (H, W), or None if all valid
        """
        if value is None:
            self._v_mask = None
            return

        if not isinstance(value, (torch.Tensor, np.ndarray)):
            raise TypeError(f'valid_mask must be torch.Tensor or np.ndarray, got {type(value)}')

        self._validate_shape(value)

        if not value.dtype == torch.bool and not value.dtype == np.bool_:
            raise ValueError(f'valid_mask must be boolean type, got {value.dtype}')

        if self._map is not None and value.shape != self._map.shape:
            raise ValueError(
                f'Shape mismatch: valid_mask {value.shape} != map {self._map.shape}')

        self.set_field(value, '_v_mask', field_type='data')

    @classmethod
    def from_tuple(cls, value: Union[Tuple[Union[np.ndarray, torch.Tensor]],
    Union[np.ndarray, torch.Tensor]]):
        """Construct MaskedMap from tuple or single map.

        Args:
            value: Input in formats:
                - Empty tuple -> return empty
                - (map,): Single map without mask
                - (map, mask): Map with mask
                - map: Single map without mask
        """
        if isinstance(value, tuple):
            if len(value) == 0:
                return cls(map=None)
            elif len(value) == 1:
                assert isinstance(value[0], (torch.Tensor, np.ndarray))
                return cls(map=value[0])
            elif len(value) == 2:
                assert isinstance(value[0], (torch.Tensor, np.ndarray))
                assert isinstance(value[1], (torch.Tensor, np.ndarray))
                return cls(map=value[0], v_mask=value[1])
            else:
                raise ValueError(f"Tuple must have length 0-2, got {len(value)}")
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            return cls(map=value)
        else:
            raise TypeError("should be Tuple of np.ndarray/torch.Tensor or np.ndarray/torch.Tensor")

    @staticmethod
    def _validate_shape(value: Union[torch.Tensor, np.ndarray]) -> None:
        """Validate input shape matches supported formats.

        Args:
            value: Input tensor/array to validate

        Raises:
            ValueError: If shape is not supported
        """
        if value.ndim not in [2, 3, 4]:
            raise ValueError(
                f'map must be 2D (H,W), 3D (B,H,W) or 4D (B,1,H,W), '
                f'got shape {value.shape}')

        if value.ndim == 4 and value.shape[1] != 1:
            raise ValueError(
                f'For 4D input, channel dimension must be 1, '
                f'got shape {value.shape}')

    @property
    def shape(self) -> Optional[tuple[int, ...]]:
        """Get spatial dimensions (H, W)."""
        if self.map is None:
            return None
        return self.map.shape[:2]

    @property
    def is_sparse(self) -> Optional[bool]:
        """Check if the map has invalid regions."""
        if self.map is None:
            return None
        if self.v_mask is None:  # dense
            return False
        return not self.v_mask.all()

    @property
    def is_blank(self) -> Optional[bool]:
        """Check if no any valid data."""
        if self.map is None:
            return None
        if self.v_mask is None:
            return False
        return not self.v_mask.any()

    def filter_valid(self, mask=None):
        """
        Returns: Valid data filtered by mask.
            - For single channel map [H,W]: returns [N]
            - For multi-channel map [H,W,C]: returns [N,C]
        """
        if self.map is None:
            return None

        if self.valid_num == 0:
            return None

        v_mask = intersect_masks(self.v_mask, mask)

        if v_mask is None:
            return self.map.reshape(-1, *self.map.shape[2:])  # preserve channel dimension

        # v_mask shape: [H,W]
        # When indexing [H,W,C] with [H,W], it automatically broadcasts
        # [H,W] True/False mask will be applied to first two dimensions
        return self.map[v_mask]

    @property
    def valid_num(self) -> Optional[int]:
        """Returns the number of valid data points."""
        if self.map is None:
            return None
        if self.v_mask is None:
            h, w = self.map.shape[:2]
            return h * w
        return self.v_mask.sum()

    def map_clone(self, invalid_value=-np.inf) -> Optional[np.ndarray]:
        """
        Clone map with invalid regions set to specified value.
        Returns None if map is not initialized.
        """
        if self.map is None:
            return None

        result = self.map.copy()
        if self.v_mask is not None:
            result[~self.v_mask] = invalid_value
        return result

    @property
    def device(self):
        """Get device of the map."""
        if self.map is None:
            return None
        if isinstance(self.map, torch.Tensor):
            return self.map.device
        elif isinstance(self.map, np.ndarray):
            return 'numpy: cpu'

    @property
    def field_type(self):
        """Get the data field type of the map.

        Returns:
            Type of the data field (torch.Tensor or np.ndarray), or None if map is not set
        """
        if self._map is None:
            return None
        return type(self.map)

    def __repr__(self) -> str:
        """Pretty print the instance."""
        repr_str = self.__class__.__name__
        repr_str += f'(shape={self.shape}, '
        repr_str += f'is_sparse={self.is_sparse}, '
        repr_str += f'field_type={self.field_type}, '
        repr_str += f'device={self.device})'
        return repr_str

    @numpy_only
    def crop(self, bbox: Tuple[int, int, int, int]) -> 'MaskedMap':
        """Crop a patch and return new instance."""
        return self._convert(
            self,
            apply_to=np.ndarray,
            func=lambda x, bx: imcrop(x, bx, padding_mode='edge'),
            bx=bbox
        )

    @numpy_only
    def resize(self, out_size: Tuple[int, int], **kwargs) -> 'MaskedMap':
        """Resize map and mask to target size."""

        if self.map is None:
            return self.new_empty()

        if self.is_sparse:
            dmap, v_mask, w_scale, _ = resize_sparse_map(
                sparse_map=self.map,
                valid_mask=self.v_mask,
                size=out_size,
                return_scale=True,
                reduction=kwargs.get('reduction', 'mean')  # default mean reduction
            )
        else:
            dmap, w_scale, _ = mmcv.imresize(
                self.map,
                size=(out_size[1], out_size[0]),
                return_scale=True,
                interpolation=kwargs.get('interpolation', 'bilinear')  # default bilinear interpolation
            )
            v_mask = np.ones(out_size, dtype=np.bool_) if self.v_mask is not None else None

        return self.new_empty(
            map=dmap,  #
            v_mask=v_mask
        )

    @numpy_only
    def flip(self, direction: str = 'horizontal') -> 'MaskedMap':
        """Flip the map and mask."""
        return self._convert(
            self,
            apply_to=np.ndarray,
            func=lambda x, dire: mmcv.imflip(x, dire),
            dire=direction
        )

    @numpy_only
    def pad(self, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]], **kwargs) -> 'MaskedMap':
        """Pad the map and mask."""
        pad_width = parse_padding(padding)

        if not kwargs.get('padding_mode', 'edge') == 'edge':
            raise NotImplementedError("only support padding mode 'edge' for maintain sparsity")

        return self._convert(
            self,
            apply_to=np.ndarray,
            func=lambda x, pw: np.pad(x, pad_width=pw, mode='edge'),  # cv2.CopyMakeBorder is not support bool array
            pw=pad_width
        )

    @numpy_only
    def unpad(self, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]]) -> 'MaskedMap':
        """Remove padding from the map and mask."""
        pad_width = parse_padding(padding)

        return self._convert(
            self,
            apply_to=np.ndarray,
            func=im_unpad,
            pad_width=pad_width
        )

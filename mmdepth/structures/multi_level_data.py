from abc import ABCMeta
from typing import Generic, TypeVar, List, Optional, Tuple, Union, Any, Literal, Sequence, Type, Callable, Iterator
from deprecated import deprecated
import numpy as np
import torch
import mmcv

from mmengine.utils import is_seq_of
from .base import BaseDataElement, numpy_only
from mmdepth.utils import resize_sparse_map, imcrop, im_unpad, parse_padding

T = TypeVar('T', np.ndarray, torch.Tensor, BaseDataElement)


class MultilevelData(BaseDataElement, Generic[T]):
    """A data structure for managing multi-level data.

    This class provides a container for managing data across multiple levels.
    Each level contains a single array/tensor/BaseDataSample. The class ensures proper
    handling of multi-level operations with automatic scale adjustment.

    The geometric transformations are categorized into:
    1. Spatial transformations (resize, crop)
    2. Flipping operations
    3. Padding operations

    Args:
        data: List of arrays/tensors/BaseDataSamples, one per level
    """

    def __init__(self, data: Optional[List[T]] = None) -> None:
        self._data: List[T] = []
        super().__init__(metainfo=None, data=data)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override attribute setting to restrict settable attributes."""
        if name in ('_metainfo_fields', '_data_fields', 'data'):
            super().__setattr__(name, value)
        elif name == '_data':
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Can only set attributes 'data', but got '{name}'")

    @property
    def data(self) -> List[T]:
        """Get the multi-level data."""
        return self._data

    @data.setter
    def data(self, value: Optional[Sequence[T]] = None) -> None:
        """Set multi-level data with type checking.

        Args:
            value: Sequence of data elements, must all be of same type

        Raises:
            TypeError: If input type is invalid
            ValueError: If input is empty
        """
        if value is None:
            return

        if not isinstance(value, (list, tuple)):
            raise TypeError(f'data must be a list/tuple, got {type(value)}')

        if not value:
            raise ValueError('Cannot set empty data')

        type_check = [
            is_seq_of(value, np.ndarray),
            is_seq_of(value, torch.Tensor),
            is_seq_of(value, BaseDataElement)
        ]
        if not any(type_check):
            raise TypeError('Elements must be array/tensor/BaseDataSample')

        self.set_field(list(value), '_data', field_type='data')

    @property
    def n_levels(self) -> int:
        """Return the number of levels."""
        return len(self.data)

    def __len__(self) -> int:
        """Return the number of levels."""
        return self.n_levels

    def __getitem__(self, idx: int) -> T:
        """Get data at specified level."""
        return self.data[idx]

    def __iter__(self) -> Iterator[T]:
        """Iterate through levels."""
        return iter(self.data)

    @property
    def shapes(self) -> List[Tuple[int, ...]]:
        """Get shapes of all levels."""
        return [item.shape for item in self.data]

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Get shape of last level."""
        if not self.data:
            return None
        return self.data[-1].shape

    def _get_spatial_scale(self, level_data: T) -> Tuple[float, float]:
        """Calculate spatial scaling factors for current level relative to base level.

        Args:
            level_data: Data at current level

        Returns:
            Tuple of (height_scale, width_scale)
        """
        base_h, base_w = self.shape[:2]
        curr_h, curr_w = level_data.shape[:2]
        return curr_h / base_h, curr_w / base_w

    # --- Spatial Transformations ---
    @numpy_only
    def resize(self, out_size: Tuple[int, int], **kwargs) -> 'MultilevelData':
        """Resize all levels with automatic scale adjustment.

        Args:
            out_size: Target size (height, width) for the base level
            **kwargs: Additional arguments for resize operation

        Returns:
            New MultilevelData instance with resized data
        """
        if not self.data:
            return self.new_empty()

        results = []
        for level_data in self.data:
            h_scale, w_scale = self._get_spatial_scale(level_data)
            level_size = (
                int(round(out_size[0] * h_scale)),
                int(round(out_size[1] * w_scale))
            )
            results.append(self._apply_resize(level_data, level_size, **kwargs))

        return self.new_empty(data=results)

    @staticmethod
    def _apply_resize(data: T, size: Tuple[int, int], **kwargs) -> T:
        """Apply resize operation based on data type."""
        if isinstance(data, np.ndarray):
            if data.dtype == np.bool_:
                return resize_sparse_map(None, valid_mask=data, size=size)
            return mmcv.imresize(data, size=(size[1], size[0]),
                                 interpolation=kwargs.get('interpolation', 'bilinear'))
        if isinstance(data, BaseDataElement):
            return data.resize(size, **kwargs)
        raise TypeError(f"Unsupported type {type(data)}")

    @numpy_only
    def crop(self, bbox: Tuple[int, int, int, int]) -> 'MultilevelData':
        """Crop all levels with scaled bbox.

        Args:
            bbox: Bounding box (x1, y1, x2, y2) for base level

        Returns:
            New MultilevelData instance with cropped data
        """
        if not self.data:
            return self.new_empty()

        results = []
        for level_data in self.data:
            h_scale, w_scale = self._get_spatial_scale(level_data)
            level_bbox = (
                int(round(bbox[0] * w_scale)),
                int(round(bbox[1] * h_scale)),
                int(round(bbox[2] * w_scale)),
                int(round(bbox[3] * h_scale))
            )
            results.append(self._apply_crop(level_data, level_bbox))

        return self.new_empty(data=results)

    @staticmethod
    def _apply_crop(data: T, bbox: Tuple[int, int, int, int]) -> T:
        """Apply crop operation based on data type."""
        if isinstance(data, np.ndarray):
            return imcrop(data, bbox, padding_mode='edge')
        if isinstance(data, BaseDataElement):
            return data.crop(bbox)
        raise TypeError(f"Unsupported type {type(data)}")

    # --- Flipping Operations ---
    @numpy_only
    def flip(self, direction: Literal['horizontal', 'vertical'] = 'horizontal') -> 'MultilevelData':
        """Flip all levels in specified direction.

        Args:
            direction: Flip direction ('horizontal' or 'vertical')

        Returns:
            New MultilevelData instance with flipped data
        """
        if not self.data:
            return self.new_empty()

        results = []
        for level_data in self.data:
            results.append(self._apply_flip(level_data, direction))

        return self.new_empty(data=results)

    @staticmethod
    def _apply_flip(data: T, direction: str) -> T:
        """Apply flip operation based on data type."""
        if isinstance(data, np.ndarray):
            return mmcv.imflip(data, direction)
        if isinstance(data, BaseDataElement):
            return data.flip(direction)
        raise TypeError(f"Unsupported type {type(data)}")

    @numpy_only
    def pad(self, padding: Union[int, Tuple]) -> 'MultilevelData':
        """Pad all levels with automatic scale adjustment.

        Args:
            padding: Padding size for base level (see _parse_padding for formats)

        Returns:
            New MultilevelData instance with padded data
        """
        if not self.data:
            return self.new_empty()

        pad_t, pad_b, pad_l, pad_r = parse_padding(padding)
        results = []

        for level_data in self.data:
            h_scale, w_scale = self._get_spatial_scale(level_data)
            level_padding = (
                int(round(pad_t * h_scale)),
                int(round(pad_b * h_scale)),
                int(round(pad_l * w_scale)),
                int(round(pad_r * w_scale))
            )
            results.append(self._apply_padding(level_data, level_padding))

        return self.new_empty(data=results)

    @staticmethod
    def _apply_padding(data: T, padding: Tuple[int, int, int, int]) -> T:
        """Apply padding operation based on data type."""
        if isinstance(data, np.ndarray):
            pad_width = ((padding[0], padding[1]), (padding[2], padding[3]))
            return np.pad(data, pad_width=pad_width, mode='edge')
        if isinstance(data, BaseDataElement):
            return data.pad(padding=padding)
        raise TypeError(f"Unsupported type {type(data)}")

    @numpy_only
    def unpad(self, padding: Union[int, Tuple]) -> 'MultilevelData':
        """Remove padding from all levels with automatic scale adjustment.

        Args:
            padding: Padding size to remove (see _parse_padding for formats)

        Returns:
            New MultilevelData instance with padding removed
        """
        if not self.data:
            return self.new_empty()

        pad_t, pad_b, pad_l, pad_r = parse_padding(padding)
        results = []

        for level_data in self.data:
            h_scale, w_scale = self._get_spatial_scale(level_data)
            level_padding = (
                int(round(pad_t * h_scale)),
                int(round(pad_b * h_scale)),
                int(round(pad_l * w_scale)),
                int(round(pad_r * w_scale))
            )
            results.append(self._apply_unpad(level_data, level_padding))

        return self.new_empty(data=results)

    @staticmethod
    def _apply_unpad(data: T, padding: Tuple[int, int, int, int]) -> T:
        """Apply unpad operation based on data type."""
        if isinstance(data, np.ndarray):
            pad_width = ((padding[0], padding[1]), (padding[2], padding[3]))
            return im_unpad(data, pad_width=pad_width)
        if isinstance(data, BaseDataElement):
            return data.unpad(padding=padding)
        raise TypeError(f"Unsupported type {type(data)}")

    def multi_apply(self, func: Callable[..., T], *args, **kwargs) -> List:
        """Apply a function to each level's data individually.

        Args:
            func: Function to apply to each level's data
            *args: Additional positional arguments passed to func
            **kwargs: Additional keyword arguments passed to func

        Returns:
            A new MultilevelData instance containing the transformed data.
            The original metainfo is not copied.
        """
        results = []
        for level_data in self.data:
            result = func(level_data, *args, **kwargs)
            results.append(result)
        return results

    def __repr__(self) -> str:
        """Return string representation."""
        main_info = f'{self.__class__.__name__} with {self.n_levels} levels'
        shape_info = 'Shapes per level:\n'

        for i, shape in enumerate(self.shapes):
            shape_info += f'  Level {i}: {shape}\n'

        if len(self.metainfo_keys()) > 0:
            shape_info += "Metainfo:\n"
            for k, v in self.metainfo_items():
                shape_info += f"    {k}: {v}\n"

        return f'{main_info}\n{shape_info}'

    # Additional utility methods
    def append(self, value: T) -> None:
        """Append new level data."""
        self.data.append(value)

    def pop(self) -> T:
        """Remove and return the last level data."""
        return self.data.pop()

    def values(self):
        """scan each data"""
        for v in self.data:
            yield v

    # Disabled methods (for clarity)
    def items(self):
        raise NotImplementedError("items() not supported for MultilevelData")

    def keys(self):
        raise NotImplementedError("keys() not supported for MultilevelData")

    def all_keys(self) -> list:
        raise NotImplementedError("all_keys() not supported for MultilevelData")

    def all_items(self):
        raise NotImplementedError("all_items() not supported for MultilevelData")

    def all_values(self) -> list:
        raise NotImplementedError("all_values() not supported for MultilevelData")

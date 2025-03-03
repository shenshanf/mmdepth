from abc import abstractmethod, ABC
from typing import Union, Optional, Tuple, Type, Callable, Any, Dict, List, Literal

import numpy as np
import torch
from mmengine.structures import BaseDataElement as MMBaseDataElement


def numpy_only(func: Callable):
    """A decorator that ensures the method only runs when data_field is numpy.ndarray.

    Args:
        func: The method to be decorated

    Raises:
        TypeError: If data_field is not numpy.ndarray
    """

    def wrapper(self: BaseDataElement, *args, **kwargs):
        for k, v in self.items():
            if not isinstance(v, np.ndarray):
                raise TypeError(
                    f"{k} must be np.ndarray when calling {func.__name__}(), "
                    f"but got {type(v).__name__}")
        return func(self, *args, **kwargs)

    return wrapper


def numpy_map_only(func):
    """A decorator that ensures the method only runs when data_field is map style numpy.ndarray.

    Args:
        func: The method to be decorated

    Raises:
        TypeError: If data_field is not numpy.ndarray
        ValueError: If numpy.ndarray does not have valid map shape (H,W) or (H,W,C) where C is 1 or 3
    """

    def wrapper(self: BaseDataElement, *args, **kwargs):
        for k, v in self.items():
            # Check type
            if not isinstance(v, np.ndarray):
                raise TypeError(
                    f"{k} must be np.ndarray when calling {func.__name__}(), "
                    f"but got {type(v).__name__}")

            # Check shape
            ndim = len(v.shape)
            if ndim == 2:
                continue  # (H,W) is valid

            if ndim == 3 and v.shape[2] in [1, 3]:
                continue  # (H,W,1) or (H,W,3) is valid

            raise ValueError(
                f"Shape of {k} must be (H,W) or (H,W,1) or (H,W,3) when calling "
                f"{func.__name__}(), but got shape {v.shape}")

        return func(self, *args, **kwargs)

    return wrapper


class BaseDataElement(MMBaseDataElement, ABC):
    """Base class for all data sample types in the framework.

    This abstract class defines the interface and common functionality for data samples
    used across different components (BaseTransform, BaseModel, Evaluator).

    Key features:
    - Automatic type handling for tensor/array data fields
    - Common data operations (crop, resize, flip, pad)
    - Device/type conversion utilities
    """

    def __setitem__(self, key: str, value: Any) -> None:
        """Set field value with automatic type detection.

        Args:
            key: Field name
            value: Field value to set (tensor/array -> data field, other -> metainfo)
        """
        field_type = 'data' if isinstance(value, (torch.Tensor, np.ndarray)) else 'metainfo'
        self.set_field(value, key, field_type=field_type)

    @property
    @abstractmethod
    def shape(self) -> Optional[Tuple[int, int]]:
        """Get shape"""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the data sample."""
        pass

    @abstractmethod
    def crop(self, bbox: Tuple[int, int, int, int]) -> 'BaseDataElement':
        """Crop the data sample given a bounding box.

        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)

        Returns:
            Cropped data sample
        """
        pass

    @abstractmethod
    def resize(self, out_size: Tuple[int, int], **kwargs) -> 'BaseDataElement':
        """Resize the data sample to target size.

        Args:
            out_size: Target size (H, W)

        Returns:
            Resized data sample
        """
        pass

    @abstractmethod
    def flip(self, direction: str = 'vertical') -> 'BaseDataElement':
        """Flip the data sample horizontally or vertically.

        Args:
            direction: Flip direction

        Returns:
            Flipped data sample
        """
        pass

    @abstractmethod
    def pad(self, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]], **kwargs) -> 'BaseDataElement':
        """Pad the data sample.

        Args:
            padding: Padding size in pixels
            kwargs: other padding config

        Returns:
            Padded data sample
        """
        pass

    @abstractmethod
    def unpad(self, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]]) -> 'BaseDataElement':
        """Remove padding from the data sample.

        Args:
            padding: Padding size to remove

        Returns:
            Unpadded data sample
        """
        pass

    def new_empty(self, copy_metainfo: bool = True, *, metainfo: Optional[Dict] = None,
                  **kwargs) -> 'BaseDataElement':
        """Create new instance with optional metadata and data fields.

        Args:
            copy_metainfo: Whether to copy original metadata
            metainfo: New metadata to update/override
            kwargs: New data fields to set

        Returns:
            New data sample instance
        """
        new_data = self.__class__()
        if copy_metainfo:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        if kwargs:
            new_data.set_data(kwargs)
        return new_data

    def _convert(self, data: Any, apply_to: Type, func: Callable, **kwargs) -> Any:
        """Convert data recursively with given function.

        Handles conversion for:
        - Tensors and arrays
        - Nested data samples
        - Lists, tuples and dictionaries

        Args:
            data: Data to convert
            apply_to: Target type to apply conversion
            func: Conversion function
            kwargs: Additional arguments for conversion function

        Returns:
            Converted data maintaining original structure

        Raises:
            TypeError: For unsupported data types
        """
        if data is None:
            return None

        if isinstance(data, (torch.Tensor, np.ndarray)):
            return func(data, **kwargs) if isinstance(data, apply_to) else data

        if isinstance(data, BaseDataElement):
            new_data = data.new_empty()
            for k, v in data.items():
                setattr(new_data, k, self._convert(v, apply_to, func, **kwargs))
            return new_data

        if isinstance(data, (List, Tuple)):
            new_data = [self._convert(item, apply_to, func, **kwargs) for item in data]
            return tuple(new_data) if isinstance(data, tuple) else new_data

        if isinstance(data, Dict):
            return {k: self._convert(v, apply_to, func, **kwargs) for k, v in data.items()}

        raise TypeError(f'Unsupported type {type(data)} for conversion')

    def cpu(self) -> 'BaseDataElement':
        """Convert all tensors to CPU."""
        return self._convert(self, torch.Tensor, lambda x: x.cpu())

    def cuda(self) -> 'BaseDataElement':
        """Convert all tensors to CUDA."""
        return self._convert(self, torch.Tensor, lambda x: x.cuda())

    def detach(self) -> 'BaseDataElement':
        """Detach all tensors from computation graph."""
        return self._convert(self, torch.Tensor, lambda x: x.detach())

    def numpy(self) -> 'BaseDataElement':
        """Convert all tensors to numpy arrays."""
        return self._convert(self, torch.Tensor, lambda x: x.detach().cpu().numpy())

    def to_tensor(self) -> 'BaseDataElement':
        """Convert all numpy arrays to tensors."""
        return self._convert(self, np.ndarray, lambda x: torch.from_numpy(x))

    def to(self, *args: Any, **kwargs: Any) -> 'BaseDataElement':
        """Move all tensors to specified device/dtype."""
        return self._convert(self, torch.Tensor, lambda x: x.to(*args, **kwargs))

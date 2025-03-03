from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union, Final
import numpy as np

from mmdepth.registry import TRANSFORMS


class BaseDispParser(ABC):
    """Base class for parsing disparity maps from different formats.

    This abstract class defines the interface for disparity map parsers that can
    convert various input formats into standardized floating-point disparity maps
    with optional validity masks.

    Args:
        invalid_value (float, optional): Value indicating invalid disparity pixels.
            Pixels with this value will be marked as invalid in the validity mask.
            If None, all pixels are considered valid and no mask will be generated.
    """

    def __init__(self, invalid_value: Optional[float] = None) -> None:
        self.invalid_value = invalid_value

    @abstractmethod
    def __call__(self, disp: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Parse raw disparity data into standardized format with validity mask.

        Args:
            disp (np.ndarray): Raw disparity map data

        Returns:
            tuple:
                - np.ndarray: Parsed disparity map as float32
                - np.ndarray or None: Boolean validity mask, None if all pixels valid

        Raises:
            ValueError: If input disparity format is invalid
        """
        pass


@TRANSFORMS.register_module()
class RawDispParser(BaseDispParser):
    """Parser for raw floating-point disparity maps.

    This parser directly handles disparity maps that are already in floating-point format,
    performing only basic validation and type conversion if needed. No transformation
    is applied to the input values.
    """

    def __call__(self, disp: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Direct parsing of float disparity maps without transformation.

        Args:
            disp (np.ndarray): Input disparity map

        Returns:
            tuple:
                - np.ndarray: Disparity map converted to float32
                - np.ndarray or None: Validity mask based on invalid_value, None if
                  invalid_value is None
        """
        # Ensure contiguous float32 array
        parsed_disp = np.ascontiguousarray(disp.astype(np.float32))

        # Return None for mask if no invalid value specified
        valid_mask = None if self.invalid_value is None else np.not_equal(parsed_disp, self.invalid_value)

        return parsed_disp, valid_mask


@TRANSFORMS.register_module()
class LinearDispParser(BaseDispParser):
    """Parser for disparity maps requiring linear transformation.

    This parser handles disparity maps that require linear transformation (scale and offset)
    to convert encoded format to actual disparity values. Supports both multiplication and
    division operations, with configurable operation order.

    Args:
        scale (float): Scale factor for transformation. Defaults to 256.0.
        offset (float): Offset value to add after scaling. Defaults to 0.0.
        operation (str): Scale operation type, either 'div' or 'mul'. Defaults to 'div'.
        invalid_value (float, optional): Value indicating invalid pixels. Defaults to 0.0.
            If None, all pixels are considered valid.
        transform_first (bool): Whether to transform values before validity check.
            Defaults to False.
    """
    VALID_OPERATIONS = {'div', 'mul'}

    def __init__(self,
                 scale: float = 256.0,
                 offset: float = 0.0,
                 operation: str = 'div',
                 invalid_value: Optional[float] = 0.0,
                 transform_first: bool = False) -> None:
        super().__init__(invalid_value)
        if operation not in self.VALID_OPERATIONS:
            raise ValueError(f"Operation must be one of {self.VALID_OPERATIONS}")

        self.scale = scale
        self.offset = offset
        self.operation = operation
        self.transform_first = transform_first

    def _apply_transform(self, disp: np.ndarray) -> np.ndarray:
        """Apply linear transformation to disparity values."""
        if self.operation == 'div':
            return (disp / self.scale) + self.offset
        return (disp * self.scale) + self.offset

    def __call__(self, disp: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Parse disparity map applying linear transformation."""
        # Ensure float32 and contiguous memory layout
        parsed_disp = np.ascontiguousarray(disp.astype(np.float32))

        # Handle case where no invalid value is specified
        if self.invalid_value is None:
            return self._apply_transform(parsed_disp), None

        if self.transform_first:
            # Transform all values then create mask
            parsed_disp = self._apply_transform(parsed_disp)
            valid_mask = np.not_equal(parsed_disp, self.invalid_value)
        else:
            # Create mask then transform valid values only
            valid_mask = np.not_equal(parsed_disp, self.invalid_value)
            parsed_disp[valid_mask] = self._apply_transform(parsed_disp[valid_mask])

        return parsed_disp, valid_mask


@TRANSFORMS.register_module()
class ColorEncodedDispParser(BaseDispParser):
    """Parser for color-encoded disparity maps.

    This parser handles disparity maps where the disparity values are encoded
    across color channels using specified weights for each channel. Commonly used
    in formats where disparity is stored as a color image.

    Args:
        weights (tuple[float, float, float]): Weights for R,G,B channels respectively.
            Defaults to (128.0, 1.0, 0.0).
        invalid_value (float, optional): Value indicating invalid pixels. Defaults to 0.0.
            If None, all pixels are considered valid.
    """

    def __init__(self,
                 weights: Tuple[float, float, float] = (128.0, 1.0, 0.0),
                 invalid_value: Optional[float] = None) -> None:
        super().__init__(invalid_value)
        self.weights: np.ndarray = np.array(weights, dtype=np.float32)

    def __call__(self, disp: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Parse color-encoded disparity map by combining weighted channels."""
        if disp.ndim != 3 or disp.shape[2] != 3:
            raise ValueError(
                f"Expected color image with shape (H,W,3), got shape {disp.shape}")

        # Apply channel weights and sum
        parsed_disp = np.sum(disp.astype(np.float32) * self.weights, axis=2)

        # Ensure contiguous memory layout
        parsed_disp = np.ascontiguousarray(parsed_disp)

        # Return None for mask if no invalid value specified
        valid_mask = None if self.invalid_value is None else np.not_equal(parsed_disp, self.invalid_value)

        return parsed_disp, valid_mask

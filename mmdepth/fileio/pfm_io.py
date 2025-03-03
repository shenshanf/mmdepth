# Copyright (c) MMDepth Contributors. All rights reserved.
#
# This code is modified from https://github.com/MartinPeris/justPFM
# which is available under the MIT License.

# Extended functionality:
# - Added support for conversion between PFM and bytes streams
# - Integrated with mmengine.fileio to enable storage of bytes data
#   across different storage backends

from math import isclose
from pathlib import Path
from sys import byteorder
from typing import Tuple, Union

from io import BytesIO

import numpy as np
import mmengine.fileio as fileio


# write pfm

def write_pfm(file_name: Union[Path, str], data: np.ndarray, scale: float = 1.0) -> None:
    """Write numpy array to PFM file.

    Args:
        file_name (Union[Path, str]): Output PFM file path
        data (np.ndarray): Input numpy array
        scale (float, optional): Scale factor. Defaults to 1.0.

    Raises:
        ValueError: If scale is 0 or negative, or data format is invalid
    """
    if isinstance(file_name, str):
        file_name = Path(file_name)

    assert file_name.suffix.lower() == '.pfm'

    # Convert data to bytes with scale
    data_bytes = bytes_from_pfm(data, scale)

    # Use fileio to write bytes to file
    fileio.put(data_bytes, file_name)


def read_pfm(file_name: Union[Path, str]) -> np.ndarray:
    """Read PFM file into numpy array.

    Args:
        file_name (Union[Path, str]): Input PFM file path

    Returns:
        np.ndarray: Decoded image data

    Raises:
        ValueError: If file format is invalid
    """
    if isinstance(file_name, str):
        file_name = Path(file_name)

    assert file_name.suffix.lower() == '.pfm'

    # Read bytes from file using fileio
    img_bytes = fileio.get(file_name)

    # Convert bytes to numpy array
    return pfm_from_bytes(img_bytes)


def pfm_from_bytes(img_bytes: Union[bytes, bytearray]) -> np.ndarray:
    """Decode PFM format from byte stream.

    Args:
        img_bytes (Union[bytes, bytearray]): Input byte stream of PFM file

    Returns:
        np.ndarray: Decoded disparity map

    Raises:
        ValueError: If the byte stream is not a valid PFM file
    """
    with BytesIO(img_bytes) as file:
        # Get number of channels from identifier
        channels = _get_pfm_channels_from_line(file.readline())

        # Get dimensions
        width, height = _get_pfm_width_and_height_from_line(file.readline())

        # Get scale and endianness
        scale, endianness = _get_pfm_scale_and_endianness_from_line(file.readline())

        # Read the data
        data = np.frombuffer(file.read(), endianness + "f")

        # Reshape and flip
        shape = (height, width, channels) if channels == 3 else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data * scale if not isclose(scale, 1.0) else data


def bytes_from_pfm(img: np.ndarray, scale: float = 1.0) -> Union[bytes, bytearray]:
    """Convert numpy array to PFM format byte stream.

    Args:
        img (np.ndarray): Input numpy array representing the image
        scale (float, optional): Scale factor. Defaults to 1.0.

    Returns:
        Union[bytes, bytearray]: PFM format byte stream

    Raises:
        ValueError: If the input array is not valid for PFM format or scale is invalid
    """
    if not _is_valid_shape(img):
        raise ValueError("data has invalid shape: " + str(img.shape))
    if img.dtype != "float32":
        raise ValueError("data must be float32: " + str(img.dtype))
    if isclose(scale, 0.0):
        raise ValueError("0 is not a valid value for scale")
    if scale < 0:
        raise ValueError("Scale must be positive, endianness is handled internally")

    with BytesIO() as bio:
        # Write identifier
        identifier = _get_pfm_identifier_from_data(img)
        bio.write(identifier.encode())

        # Write dimensions
        width, height = _get_pfm_width_and_height_from_data(img)
        bio.write(f"\n{width} {height}\n".encode())

        # Write scale and endianness
        # Multiply user scale with endianness indicator
        final_scale = scale * _get_pfm_endianness_from_data(img)
        bio.write(f"{final_scale}\n".encode())

        # Write data (scaled if needed)
        flipped_data = np.flipud(img)
        if not isclose(scale, 1.0):
            flipped_data = flipped_data * scale
        flipped_data.tofile(bio)

        return bio.getvalue()


def _get_pfm_identifier_from_data(data: np.ndarray) -> str:
    """Get the pfm identifier depending on the number of channels on the
    data object
    """
    identifier = "Pf"
    if len(data.shape) == 3 and data.shape[2] == 3:
        identifier = "PF"
    return identifier


def _is_valid_shape(data: np.ndarray) -> bool:
    """Return true if the shape of the data is valid"""
    if len(data.shape) == 2:
        return True

    if len(data.shape) == 3:
        if data.shape[2] == 1 or data.shape[2] == 3:
            return True
    return False


def _get_pfm_width_and_height_from_data(data: np.ndarray) -> Tuple[int, int]:
    """Return the width and height of the matrix in the proper order"""
    height, width = data.shape[:2]
    return width, height


def _get_pfm_endianness_from_data(data: np.ndarray) -> float:
    """Return 1 if bigendian, -1 if little endian data"""
    endianness = data.dtype.byteorder
    return (
        -1 if endianness == "<" or (endianness == "=" and byteorder == "little") else 1
    )


def _get_pfm_channels_from_line(line: bytes) -> int:
    """Returns the number of channels of the data based on the PFM identifier"""
    identifier = line.rstrip().decode("UTF-8")
    channels = 0
    if identifier == "Pf":
        channels = 1
    elif identifier == "PF":
        channels = 3
    else:
        raise ValueError("Not a valid PFM identifier")
    return channels


def _get_pfm_width_and_height_from_line(line: bytes) -> Tuple[int, int]:
    """Parses the width and height from the PFM header"""
    decoded_line = line.rstrip().decode("UTF-8")
    items = decoded_line.split()
    if len(items) == 2:
        width = int(items[0])
        height = int(items[1])
    else:
        raise ValueError("Not a valid PFM header")
    return width, height


def _get_pfm_scale_and_endianness_from_line(line: bytes) -> Tuple[float, str]:
    """Parse the scale and endianness from the PFM header"""
    decoded_line = line.rstrip().decode("UTF-8")
    scale = float(decoded_line)
    if isclose(scale, 0.0):
        raise ValueError("0 is not a valid value for scale")
    endianness = ""
    if scale < 0:
        endianness = "<"
        scale = -scale
    else:
        endianness = ">"
    return scale, endianness

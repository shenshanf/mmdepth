from typing import Optional, Union, Tuple, Sequence, Any
import numpy as np


def resize_sparse_map(
        sparse_map: Optional[np.ndarray],
        valid_mask: np.ndarray,
        size: Tuple[int, int],
        reduction: str = 'mean',
        return_scale: bool = False
):
    """Resize sparse map and valid mask. Compare with cv2.resize with nearest interpolation,
    this method will consider all valid value.

    Args:
        sparse_map: Optional dense format map with shape (H, W).
        valid_mask: Boolean mask indicating valid positions with shape (H, W).
        size: Target size (new_H, new_W).
        reduction: Reduction method for overlapping positions.
                  Options: ['last', 'mean', 'min', 'max'].
        return_scale: Whether to return scaling factors.

    Returns:
        If sparse_map is None:
            - return_scale=False: resized valid_mask
            - return_scale=True: (resized valid_mask, scale factors)
        If sparse_map is not None:
            - return_scale=False: (resized sparse_map, resized valid_mask)
            - return_scale=True: (resized sparse_map, resized valid_mask, scale factors)
    """
    # Get original and target sizes
    h, w = valid_mask.shape
    new_h, new_w = size
    scale_h, scale_w = h / new_h, w / new_w

    # Handle case with no valid points
    if not valid_mask.any():
        resized_mask = np.zeros(size, dtype=bool)
        if sparse_map is None:
            return (resized_mask, scale_h, scale_w) if return_scale else resized_mask
        result = np.zeros(size, dtype=sparse_map.dtype)
        return (result, resized_mask, scale_h, scale_w) if return_scale else (result, resized_mask)

    # Get valid positions and calculate new coordinates
    valid_y, valid_x = np.where(valid_mask)
    new_y = (valid_y / scale_h).astype(int)
    new_x = (valid_x / scale_w).astype(int)

    # Resize valid mask using logical_or for overlapping positions
    resized_mask = np.zeros(size, dtype=bool)
    np.logical_or.at(resized_mask, (new_y, new_x), True)

    # Return only resized mask if no sparse map provided
    if sparse_map is None:
        return (resized_mask, scale_h, scale_w) if return_scale else resized_mask

    # Resize sparse map based on reduction method
    result = np.zeros(size, dtype=sparse_map.dtype)
    values = sparse_map[valid_y, valid_x]

    if reduction == 'last':
        # Simply override with last value for each position
        result[new_y, new_x] = values
    elif reduction == 'max':
        # Take maximum of overlapping values
        np.maximum.at(result, (new_y, new_x), values)
    elif reduction == 'min':
        # Take minimum of overlapping values
        result.fill(np.inf)
        np.minimum.at(result, (new_y, new_x), values)
        result[result == np.inf] = 0
    elif reduction == 'mean':
        # Calculate mean of overlapping values
        count = np.zeros(size, dtype=np.int32)
        np.add.at(count, (new_y, new_x), 1)
        np.add.at(result, (new_y, new_x), values)
        mask = count > 0
        result[mask] /= count[mask]

    # Return results based on return_scale flag
    if return_scale:
        return result, resized_mask, scale_h, scale_w
    return result, resized_mask


def imcrop(img: np.ndarray, bbox: tuple, padding_mode: str = 'edge', **kwargs: Any) -> np.ndarray:
    """Crop image with automatic padding when crop region exceeds image boundaries."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # Clip crop coordinates to valid image region
    valid_x1 = max(0, x1)
    valid_y1 = max(0, y1)
    valid_x2 = min(w - 1, x2)
    valid_y2 = min(h - 1, y2)

    # Calculate required padding
    pad_top = abs(min(0, y1))
    pad_bottom = max(0, y2 - h + 1)
    pad_left = abs(min(0, x1))
    pad_right = max(0, x2 - w + 1)

    # Crop valid region
    cropped = img[valid_y1:valid_y2 + 1, valid_x1:valid_x2 + 1].copy()

    # Add padding if needed
    if any((pad_top, pad_bottom, pad_left, pad_right)):
        # Convert pad_width to tuple of tuples for np.pad
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        if img.ndim == 3:
            pad_width = (*pad_width, (0, 0))

        cropped = np.pad(cropped, pad_width, mode=padding_mode, **kwargs)

    return cropped


def im_unpad(img: np.ndarray,
             pad_width: Union[Tuple[Tuple[int, int], Tuple[int, int]], Sequence[Sequence[int]]]) -> np.ndarray:
    """Remove padding from image array.

    Args:
        img: Input image array with shape (H, W) or (H, W, C)
        pad_width: Number of pixels padded to each dimension, in np.pad format:
            ((top, bottom), (left, right))

    Returns:
        np.ndarray: Un_padded image array

    Raises:
        ValueError: If padding values exceed image dimensions
    """
    if len(pad_width) < 2:
        raise ValueError('pad_width must specify padding for at least height and width')

    (top, bottom), (left, right) = pad_width[:2]
    h, w = img.shape[:2]

    # Check padding validity
    if top >= h or bottom >= h or left >= w or right >= w:
        raise ValueError('Padding values exceed image dimensions')

    if top + bottom >= h or left + right >= w:
        raise ValueError('Invalid padding values: would result in empty image')

    # Compute valid region
    y1, y2 = top, h - bottom
    x1, x2 = left, w - right

    # Extract un_padded region
    if img.ndim == 3:
        return img[y1:y2, x1:x2, :].copy()
    return img[y1:y2, x1:x2].copy()

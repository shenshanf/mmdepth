from typing import List, Optional, Union
import numpy as np
import torch


def equal_length(*args):
    """

    Args:
        *args:

    Returns:

    """
    if not args:
        return True

    first_length = len(args[0])
    return all(len(lst) == first_length for lst in args[1:])


def equal_nonempty_length(*args):
    """
    Check if multiple sequences are non-empty and have equal length
    """
    # Return True if no arguments provided
    if not args:
        return True

    # Check if any sequence is empty
    if any(not lst for lst in args):
        return False

    # Get the length of first sequence
    first_length = len(args[0])

    # Check if all sequences have the same length
    return all(len(lst) == first_length for lst in args[1:])


def parse_padding(padding):
    """parse padding to 'numpy pad' support format"""
    if isinstance(padding, int):
        pad_width = ((padding, padding), (padding, padding))
    elif isinstance(padding, tuple):
        if len(padding) == 1:
            pad_width = ((padding[0], padding[0]), (padding[0], padding[0]))
        elif len(padding) == 2:
            pad_width = ((padding[0], padding[0]), (padding[1], padding[1]))
        elif len(padding) == 4:
            pad_width = ((padding[0], padding[1]), (padding[2], padding[3]))
        else:
            raise ValueError(f'Invalid padding format: {padding}')
    else:
        raise TypeError(f'Invalid padding type: {type(padding)}')

    return pad_width


def check_symmetric(results: dict, check_keys: List[str] = ['imgs', 'gt_disps', 'pred_disps', 'occ_masks']) -> bool:
    """Check if the specified keys in results contain symmetric data pairs.

    Args:
        results (dict): Dictionary containing data to check
        check_keys (List[str]): List of keys to verify for symmetry

    Returns:
        bool: True if all specified keys contain valid symmetric pairs,
              False otherwise

    Notes:
        A valid symmetric pair must:
        1. Be a list/tuple of length 2
        2. Have both elements being non-None
    """
    if not isinstance(results, dict):
        return False

    for key in check_keys:
        # Skip if key doesn't exist
        if key not in results:
            continue

        data_pair = results[key]

        # Validate data type and length
        if not isinstance(data_pair, (list, tuple)):
            return False
        if len(data_pair) != 2:
            return False

        # Check if both elements exist
        if data_pair[0] is None or data_pair[1] is None:
            return False

    return True


def intersect_masks(*masks: Optional[Union[np.ndarray, torch.Tensor]]) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """Calculate intersection of multiple masks efficiently.

    Args:
        *masks: Variable number of boolean masks or None.
                None is treated as all-True mask.

    Returns:
        Optional[np.ndarray]: Intersection of all masks.
        Returns None if all inputs are None (treated as dense case).
    """
    # Filter out None masks
    valid_masks = [mask for mask in masks if mask is not None]

    # If all masks are None, return None (dense case)
    if not valid_masks:
        return None

    # If only one valid mask, return it directly
    if len(valid_masks) == 1:
        return valid_masks[0]

    # Calculate intersection of all valid masks
    result = valid_masks[0]
    for mask in valid_masks[1:]:
        result = result & mask

    return result

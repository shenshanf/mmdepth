from typing import Dict, Optional, Union, Tuple, List, Sequence, Any

import numpy as np
import torch
import mmcv
import mmengine
from mmcv.transforms import Normalize as MM_Normalize
from mmcv.transforms import BaseTransform
from mmengine.logging import print_log

from mmdepth.registry import TRANSFORMS

Number = Union[int, float]


@TRANSFORMS.register_module()
class Normalize(MM_Normalize):
    """Normalize images with given mean and standard deviation.

    This transform inherits from mmcv's Normalize transform and adds support
    for processing multiple images.

    Args:
        mean (Sequence[Number]): Mean values for each channel.
        std (Sequence[Number]): Standard deviation values for each channel.
        to_rgb (bool): Whether to convert to RGB format. Defaults to True.

    Note:
        The channel order matters. It should match the original image format.
        For BGR image, mean and std should be in BGR order.
    """

    def __init__(self,
                 mean: Sequence[Number],
                 std: Sequence[Number],
                 to_rgb: bool = True) -> None:
        super().__init__(mean, std, to_rgb)

    def transform(self, results: Dict) -> Dict:
        """Transform function to normalize images.
        """
        # Check imgs exists
        assert 'imgs' in results, "'imgs' key must exist in results dict"

        # Skip if already normalized
        if results.get('img_norm_cfg', None) is not None:
            print_log("already normalized, and skip", level=30)
            return results

        imgs = []
        for img in results['imgs']:
            imgs.append(mmcv.imnormalize(img, self.mean, self.std, self.to_rgb))

        results['imgs'] = imgs
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results


@TRANSFORMS.register_module()
class Denormalize(BaseTransform):
    """Denormalize images by applying inverse normalization.

    This transform can either:
    1. Use all None parameters to auto-use configs from img_norm_cfg
    2. Use all manually specified parameters for denormalization

    Args:
        mean (Sequence[Number], optional): Mean values for each channel.
        std (Sequence[Number], optional): Standard deviation values for each channel.
        to_bgr (bool, optional): Whether to convert back to BGR format.

    Note:
        If any parameter is None, all parameters must be None and img_norm_cfg
        must exist in results dict.
        If any parameter is set, all parameters must be set manually.
    """

    def __init__(self,
                 mean: Optional[Sequence[Number]] = None,
                 std: Optional[Sequence[Number]] = None,
                 to_bgr: Optional[bool] = None) -> None:
        super().__init__()
        # Validate parameters consistency
        params = [mean, std, to_bgr]
        if any(p is None for p in params):
            assert all(p is None for p in params), (
                'All parameters must be None or all parameters must be set. '
                'Mixed None and non-None parameters are not allowed.')
            self.use_cfg = True
        else:
            self.use_cfg = False
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)
            self.to_bgr = to_bgr

    def transform(self, results: Dict) -> Dict:
        """Transform function to denormalize images.
        """
        # Check imgs exists
        assert 'imgs' in results, "'imgs' key must exist in results dict"

        # Skip if already denormalized
        if results.get('img_denorm_cfg', None) is not None:
            print_log('already denormalized, and skip', level=30)
            return results

        # Get normalization parameters
        if self.use_cfg:
            assert 'img_norm_cfg' in results, (
                'img_norm_cfg must exist when using auto config mode')
            norm_cfg = results['img_norm_cfg']
            mean = norm_cfg['mean']
            std = norm_cfg['std']
            to_bgr = not norm_cfg['to_rgb']  # Inverse of original to_rgb
        else:
            mean = self.mean
            std = self.std
            to_bgr = self.to_bgr

        imgs = []
        for img in results['imgs']:
            imgs.append(mmcv.imdenormalize(img, mean, std, to_bgr))

        results['imgs'] = imgs
        results['img_denorm_cfg'] = dict(mean=mean, std=std, to_bgr=to_bgr)

        return results

    def __repr__(self) -> str:
        """Return string representation."""
        repr_str = self.__class__.__name__
        if self.use_cfg:
            repr_str += '(use_cfg=True)'
        else:
            repr_str += f'(mean={self.mean.tolist()}, '
            repr_str += f'std={self.std.tolist()}, '
            repr_str += f'to_bgr={self.to_bgr})'
        return repr_str


def to_tensor(data: Any) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif hasattr(data, 'to_tensor'):  # note: modify from mmcv.to_tensor
        return data.to_tensor()
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class ToTensor(BaseTransform):
    """Convert data items to torch.Tensor by given keys.

    This transform handles various data types:
    - numpy arrays
    - torch tensors
    - sequence types (list/tuple)
    - scalar values

    Args:
        keys: Keys for items to be converted.
            Can be a single key or list of keys.

    Note:
        - Original data type should be convertible to torch.Tensor
        - For numpy arrays, maintains the data type after conversion
        - Other types will be converted to torch.FloatTensor
    """

    def __init__(self, keys: Union[str, Sequence[str]]) -> None:
        self.keys = [keys] if isinstance(keys, str) else list(keys)

    def transform(self, results: dict) -> dict:
        """Convert data items to torch.Tensor.

        Args:
            results (dict): Results dict containing items to be converted.
                Must contain all keys specified in self.keys.

        Returns:
            dict: Results with items converted to torch.Tensor.
                Original items are replaced with tensors.

        Raises:
            KeyError: If any specified key is missing in results.
            TypeError: If data cannot be converted to tensor.
        """
        for key in self.keys:
            if key not in results:
                raise KeyError(f'Key {key} not found in results')

            try:
                results[key] = to_tensor(results[key])
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f'Failed to convert {key} to tensor: {str(e)}. '
                    f'Data type: {type(results[key])}')

        return results

    def __repr__(self) -> str:
        """Return string representation."""
        return f'{self.__class__.__name__}(keys={self.keys})'


@TRANSFORMS.register_module()
class ImageToTensor(BaseTransform):
    """Convert image to torch.Tensor by given keys.

    This transform handles both grayscale and color images:
    - For grayscale images ([H,W] or [H,W,1]), can optionally expand to 3 channels
    - For color images ([H,W,3]), keeps the original channels
    - Converts to torch.Tensor with shape [C,H,W]

    Args:
        keys : Keys for images to be converted.
        to_color (bool): Whether to expand grayscale to 3 channels. Defaults to True.
            Only affects grayscale images, color images keep original channels.
    """

    def __init__(self, keys: Union[str, Sequence[str]], to_color: bool = True) -> None:
        self.keys = [keys] if isinstance(keys, str) else list(keys)
        self.to_color = to_color

    def transform(self, results: dict) -> dict:
        """Convert images in results to torch.Tensor.

        Args:
            results (dict): Results dict containing images to be converted.
                Must contain all keys specified in self.keys.

        Returns:
            dict: Results with converted tensor images.
                Original images are replaced with tensors.

        Raises:
            KeyError: If any specified key is missing in results.
            ValueError: If input image has invalid dimensions.
            TypeError: If input image is not numpy array.
        """
        for key in self.keys:
            if key not in results:
                raise KeyError(f'Key {key} not found in results')

            img = results[key]
            if not isinstance(img, np.ndarray):
                raise TypeError(f'Image {key} must be numpy array')

            # Handle different input dimensions
            if len(img.shape) == 2:  # [H,W]
                img = np.expand_dims(img, -1)  # Add channel dim: [H,W,1]
            elif len(img.shape) == 3:  # [H,W,C]
                if img.shape[2] not in [1, 3]:
                    raise ValueError(f'Image {key} must have 1 or 3 channels, '
                                     f'got {img.shape[2]}')
            else:
                raise ValueError(f'Image {key} must be 2D or 3D array, '
                                 f'got shape {img.shape}')

            # Expand grayscale to color if requested
            # Do this before memory optimization since it changes array layout
            if self.to_color and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)  # [H,W,1] -> [H,W,3]

            # note：
            #   Optimize memory layout and convert to tensor
            #   If array is not C-contiguous, transpose before tensor conversion
            #   to avoid extra memory copy during permute
            #   this is borrowed from mmdetection: https://github.com/open-mmlab/mmdetection
            if not img.flags.c_contiguous:
                # Transpose and ensure C-contiguous memory layout
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                results[key] = to_tensor(img)  # [C,H,W]
            else:
                # Keep original layout and use permute
                results[key] = to_tensor(img).permute(2, 0, 1).contiguous()

        return results

    def __repr__(self) -> str:
        """Return string representation."""
        return f'{self.__class__.__name__}(keys={self.keys}, to_color={self.to_color})'

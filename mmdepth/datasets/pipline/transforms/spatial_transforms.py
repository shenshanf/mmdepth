from typing import Union, Tuple, Optional, Dict, List
import math
import numpy as np

import mmcv
import cv2
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmdepth.utils import parse_padding, resize_sparse_map, im_unpad, imcrop, check_symmetric
from mmdepth.registry import TRANSFORMS


def _fixed_scale_size(
        size: Tuple[int, int],
        scale: Union[float, int, tuple],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.
    note: this function is copy from mmdetection: https://github.com/open-mmlab/mmdetection

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    # don't need o.5 offset
    return int(w * float(scale[0])), int(h * float(scale[1]))


def rescale_size(old_size: tuple,
                 scale: Union[float, int, tuple],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.
    note: this function is copy from mmdetection: https://github.com/open-mmlab/mmdetection

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')
    # only change this
    new_size = _fixed_scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(
        img: np.ndarray,
        scale: Union[float, Tuple[int, int]],
        return_scale: bool = False,
        interpolation: str = 'bilinear',
        backend: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Resize image while keeping the aspect ratio.
    note: this function is copy from mmdetection: https://github.com/open-mmlab/mmdetection

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = mmcv.imresize(
        img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


# ===


@TRANSFORMS.register_module()
class Resize(BaseTransform):
    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
                 keep_ratio: bool = False,
                 backend: str = 'cv2',
                 interpolation='bilinear',
                 reduction='mean') -> None:
        """
        Args:
            scale (int or tuple): Images scales for resizing. Defaults to None
            scale_factor (float or tuple[float]): Scale factors for resizing.
                Defaults to None.
            keep_ratio (bool): Whether to keep the aspect ratio when resizing the
                image. Defaults to False.
            backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
                These two backends generates slightly different results. Defaults
                to 'cv2'.
            interpolation (str): Interpolation method, accepted values are
                "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
                backend, "nearest", "bilinear" for 'pillow' backend. Defaults
                to 'bilinear'.
            reduction (str): reduction for interpolation sparse map when point interpolation collision
                             'mean', 'max', 'min'
        """
        super().__init__()
        self.scale = scale
        self.scale_factor = scale_factor
        self.keep_ratio = keep_ratio
        self.backend = backend
        self.interpolation = interpolation
        self.reduction = reduction

    def _resize_imgs(self, results: dict):
        """resize imgs"""

        if results.get('imgs', None) is None:
            return

        imgs = list()
        img_shape = None
        scale_factor = None
        for img in results['imgs']:
            img_shape = img.shape[:2]
            if self.keep_ratio:
                result_img, scale_factor = imrescale(img, results['scale'],
                                                     interpolation=self.interpolation,
                                                     return_scale=True,
                                                     backend=self.backend)
            else:
                result_img, w_scale, h_scale = mmcv.imresize(img, results['scale'],
                                                             interpolation=self.interpolation,
                                                             return_scale=True,
                                                             backend=self.backend)
                scale_factor = (w_scale, h_scale)
            imgs.append(result_img)

        # set result
        results['imgs'] = imgs
        results['img_shape'] = img_shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_disps(self, results: Dict, key: str = 'gt_disps'):
        """resize with datasample['scale']"""
        if results.get(key, None) is None:
            return

        disps = list()
        for disp in results[key]:
            assert hasattr(disp, 'resize')
            scale = results['scale']
            disps.append(disp.resize(out_size=scale[::-1],
                                     interpolation=self.interpolation,
                                     reduction=self.reduction))
        results[key] = disps

    def _resize_occ_masks(self, data_sample):
        """resize with datasample['scale']"""
        if data_sample.get('occ_masks', None) is None:
            return

        occ_masks = list()
        for occ_mask in data_sample['occ_masks']:
            scale = data_sample['scale']
            occ_masks.append(resize_sparse_map(None, occ_mask,
                                               size=scale[::-1], reduction=self.reduction))
        data_sample['occ_masks'] = occ_masks

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """
        keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)

        modify keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)

            - img_shape
        add keys:
            - scale
            - scale_factor
            - keep_ratio
        """

        if 'imgs' not in results.keys():
            return results

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['imgs'][0].shape[:2]
            results['scale'] = _fixed_scale_size(img_shape[::-1],
                                                 self.scale_factor)

        self._resize_imgs(results)
        self._resize_disps(results, key='gt_disps')
        self._resize_disps(results, key='pred_disps')
        self._resize_occ_masks(results)
        return results


@TRANSFORMS.register_module()
class RandomResize(BaseTransform):
    """Randomly resize images and their associated data.

    This transform randomly scales the input images and associated data (like disparity maps)
    within specified size bounds. It supports both uniform scaling and random stretching
    (non-uniform scaling in height and width).

    The scaling process works in two steps:
    1. Apply a base scale factor sampled from a log-uniform distribution
    2. Optionally apply additional random stretching to height and width independently

    The final dimensions are clipped to ensure they stay within min_size and max_size bounds.

    Args:
        min_size (Tuple[int, int]): Minimum output size (H, W). Defaults to (256, 256).
        max_size (Tuple[int, int]): Maximum output size (H, W). Defaults to (512, 512).
        stretch_prob (float): Probability of applying random stretching. Defaults to 0.4.
        min_scale (float): Minimum log2 scale factor. Defaults to -0.2 (0.87x).
        max_scale (float): Maximum log2 scale factor. Defaults to 0.4 (1.32x).
        max_stretch (float): Maximum log2 stretch factor. Defaults to 0.2 (±1.15x).
        interpolation (str): Interpolation method for resizing. Defaults to 'bilinear'.
        reduction (str): Reduction method for handling overlapping points. Defaults to 'mean'.
        backend (str): Backend for image resizing. Defaults to 'cv2'.

    Note:
        Scale factors are sampled in log2 space to ensure uniform distribution of scales.
        For example, min_scale=-0.2 means 2^(-0.2) ≈ 0.87x scaling.
    """

    def __init__(
            self,
            min_size: Tuple[int, int] = (256, 256),
            max_size: Tuple[int, int] = (512, 512),
            stretch_prob: float = 0.4,
            min_scale: float = -0.2,  # 2^(-0.2) ≈ 0.87x
            max_scale: float = 0.4,  # 2^0.4 ≈ 1.32x
            max_stretch: float = 0.2,  # 2^(±0.2) ≈ ±1.15x
            interpolation: str = 'bilinear',
            reduction: str = 'mean',
            backend: str = 'cv2'
    ) -> None:
        """Initialize RandomResize transform."""
        super().__init__()
        self._validate_params(min_size, max_size, stretch_prob,
                              min_scale, max_scale, max_stretch)

        self.min_size = min_size
        self.max_size = max_size
        self.stretch_prob = stretch_prob
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_stretch = max_stretch

        self.interpolation = interpolation
        self.reduction = reduction
        self.backend = backend

    @staticmethod
    def _validate_params(
            min_size: Tuple[int, int],
            max_size: Tuple[int, int],
            stretch_prob: float,
            min_scale: float,
            max_scale: float,
            max_stretch: float
    ) -> None:
        """Validate initialization parameters.

        Args:
            min_size: Minimum size bounds
            max_size: Maximum size bounds
            stretch_prob: Stretching probability
            min_scale: Minimum scale factor (log2)
            max_scale: Maximum scale factor (log2)
            max_stretch: Maximum stretch factor (log2)

        Raises:
            ValueError: If any parameters are invalid
        """
        if not (0 <= stretch_prob <= 1):
            raise ValueError(f'stretch_prob must be in [0, 1], got {stretch_prob}')

        if not (min_size[0] <= max_size[0] and min_size[1] <= max_size[1]):
            raise ValueError(
                f'min_size {min_size} must be <= max_size {max_size} for both dimensions')

        if min_scale > max_scale:
            raise ValueError(
                f'min_scale {min_scale} must be <= max_scale {max_scale}')

        if max_stretch < 0:
            raise ValueError(f'max_stretch must be non-negative, got {max_stretch}')

    @cache_randomness
    def _get_random_size(self, h: int, w: int) -> Tuple[int, int]:
        """Calculate random output size based on input dimensions.

        The calculation follows these steps:
        1. Sample base scale factor in log2 space
        2. Optionally apply random stretching to height and width
        3. Apply scales to input dimensions
        4. Clip results to min/max bounds

        Args:
            h: Input height
            w: Input width

        Returns:
            Tuple[int, int]: Random target size (H, W)
        """
        # Sample base scale in log2 space for uniform scaling distribution
        base_scale = 2.0 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_h = scale_w = base_scale

        # Apply random stretching with probability stretch_prob
        if np.random.random() < self.stretch_prob:
            # Sample stretch factors in log2 space
            h_stretch = 2.0 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            w_stretch = 2.0 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_h *= h_stretch
            scale_w *= w_stretch

        # Calculate target dimensions
        target_h = int(math.ceil(h * scale_h))
        target_w = int(math.ceil(w * scale_w))

        # Clip to allowed size bounds
        target_h = self._clip_value(target_h, self.min_size[0], self.max_size[0])
        target_w = self._clip_value(target_w, self.min_size[1], self.max_size[1])

        return target_h, target_w

    @staticmethod
    def _clip_value(value: int, min_val: int, max_val: int) -> int:
        """Clip value to specified range.

        Args:
            value: Input value
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            int: Clipped value
        """
        return max(min(value, max_val), min_val)

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Apply random resize transform to data sample.
        keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)

        modify keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)

            - img_shape
        add keys:
            - scale
            - scale_factor
            - keep_ratio
        """
        if 'imgs' not in results.keys():
            return results

        target_h, target_w = self._get_random_size(results['imgs'][0].shape[:2])

        resize_transform = Resize(
            scale=(target_w, target_h),
            backend=self.backend,
            interpolation=self.interpolation,
            reduction=self.reduction
        )

        return resize_transform(results)


class Crop(BaseTransform):
    """Transform that crops images and associated data to a specified region.

    This transform handles cropping of:
    - Images
    - Disparity maps (both single and multi-level)
    - Occlusion masks

    The transform supports padding when the crop region extends beyond image boundaries.

    Args:
        bbox (Tuple[int, int, int, int]): Crop region as (x1, y1, x2, y2).
        padding_mode (str): Padding method for handling regions outside image.
            Must be one of: ['constant', 'edge', 'reflect', 'symmetric'].
            Defaults to 'edge'.
    """

    def __init__(
            self,
            bbox: Tuple[int, int, int, int],
            padding_mode: str = 'edge'
    ) -> None:
        """Initialize Crop transform.

        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            padding_mode: Method for padding out-of-bounds regions

        Raises:
            ValueError: If bbox coordinates are invalid
            ValueError: If padding_mode is not supported
        """
        super().__init__()
        self.bbox = bbox
        self.padding_mode = padding_mode

    def _crop_imgs(self, results: Dict) -> None:
        """Crop images in data sample.
        """
        if results.get('imgs', None) is None:
            return

        cropped_imgs: List[np.ndarray] = []
        for img in results['imgs']:
            cropped_img = imcrop(
                img,
                self.bbox,
                padding_mode=self.padding_mode
            )
            cropped_imgs.append(cropped_img)

        results['imgs'] = cropped_imgs
        results['crop_size'] = (self.bbox[3] - self.bbox[2],
                                self.bbox[2] - self.bbox[0])
        results['img_shape'] = results['crop_size']

    def _crop_disps(self, results: Dict, key: str = 'gt_disps') -> None:
        """Crop disparity maps in data sample.
        """
        if results.get(key, None) is None:
            return

        cropped_disps = []
        for disp in results[key]:
            assert hasattr(disp, 'crop')
            cropped_disp = disp.crop(self.bbox)
            cropped_disps.append(cropped_disp)

        results[key] = cropped_disps

    def _crop_occ_masks(self, results: Dict) -> None:
        """Crop occlusion masks in data sample.
        """
        if results.get('occ_masks', None) is None:
            return

        cropped_masks = []
        for mask in results['occ_masks']:
            cropped_mask = imcrop(mask, self.bbox,
                                  padding_mode=self.padding_mode)
            cropped_masks.append(cropped_mask)

        results['occ_masks'] = cropped_masks

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Apply crop transform to data sample.
        keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)

        modify keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)

            - img_shape
        add keys:
            - crop_size
        """
        self._crop_imgs(results)
        self._crop_disps(results, key='gt_disps')
        self._crop_disps(results, key='pred_disps')
        self._crop_occ_masks(results)

        return results


class RandomCrop(BaseTransform):
    """Randomly crop images and associated data to specified size.

    This transform generates random crop coordinates within the input image
    dimensions while ensuring the crop size requirements are met.

    Args:
        crop_size (Tuple[int, int]): Target crop size as (height, width)
        padding_mode (str): Padding method for out-of-bounds regions.
            Must be one of: ['constant', 'edge', 'reflect', 'symmetric'].
            Defaults to 'edge'.
    """

    def __init__(
            self,
            crop_size: Tuple[int, int],
            padding_mode: str = 'edge'
    ) -> None:
        """Initialize RandomCrop transform.

        Args:
            crop_size: Desired crop dimensions (height, width)
            padding_mode: Method for padding out-of-bounds regions
        """
        super().__init__()
        self.crop_size = crop_size
        self.padding_mode = padding_mode

    @cache_randomness
    def _random_bbox(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """Generate random crop box coordinates.

        Ensures the crop box fits within image dimensions when possible,
        with random positioning.

        Args:
            h: Image height
            w: Image width

        Returns:
            Tuple[int, int, int, int]: Crop coordinates (x1, y1, x2, y2)
        """
        # Calculate maximum valid starting positions
        margin_h = max(0, h - self.crop_size[0])
        margin_w = max(0, w - self.crop_size[1])

        # Random starting point
        y1 = np.random.randint(0, margin_h + 1)
        x1 = np.random.randint(0, margin_w + 1)

        # Calculate end points
        y2 = y1 + self.crop_size[0]
        x2 = x1 + self.crop_size[1]

        return x1, y1, x2, y2

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Apply random crop transform to data sample.
        keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)

        modify keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)

            - img_shape
        add keys:
            - crop_size
        """
        # Ensure required data exists
        assert results.get('imgs') is not None, \
            'RandomCrop requires img_left to be present'

        # Generate random crop coordinates
        bbox = self._random_bbox(results['imgs'][0].shape[:2])

        # Apply crop transform
        crop_transform = Crop(
            bbox=bbox,
            padding_mode=self.padding_mode
        )
        return crop_transform(results)


class Flip(BaseTransform):
    """Transform that flips stereo data horizontally or vertically.

    For horizontal flips:
    - Only works with symmetric data (both left and right views present)
    - Swaps left and right data after flipping to maintain correct stereo order
    """

    def __init__(self, direction='vertical'):
        """Initialize flip transform.

        Args:
            direction (str): Flip direction, either 'horizontal' or 'vertical'.
                           Defaults to 'vertical'.
        """
        super().__init__()
        assert direction in ['horizontal', 'vertical']
        self.direction = direction

    def _flip_imgs(self, data_sample, swap=False):
        """Flip images in the data sample.

        Args:
            data_sample: Data sample containing images to flip
            swap (bool): Whether to swap left and right images after flipping.
                        Only used for horizontal flips.
        """
        if data_sample.get('imgs', None) is None:
            return

        imgs = list()
        for img in data_sample['imgs']:
            imgs.append(mmcv.imflip(img, self.direction))

        if swap:
            assert len(imgs) == 2
            imgs.reverse()

        data_sample['imgs'] = imgs
        data_sample['do_flip'] = True
        data_sample['flip_direction'] = self.direction

    def _flip_disps(self, data_sample, swap, key='gt_disps'):
        """Flip disparity maps in the data sample.

        Args:
            data_sample: Data sample containing disparity maps
            swap (bool): Whether to swap left and right disparity maps
            key (str): Key for accessing disparity data ('gt_disps' or 'pred_disps').
                      Defaults to 'gt_disps'.
        """
        if data_sample.get(key, None) is None:
            return

        disps = list()
        for disp in data_sample[key]:
            assert hasattr(disp, 'flip')
            disps.append(disp.flip(self.direction))

        if swap:
            assert len(disps) == 2
            disps.reverse()
        data_sample[key] = disps

    def _flip_occ_masks(self, data_sample, swap):
        """Flip occlusion masks in the data sample.

        Args:
            data_sample: Data sample containing occlusion masks
            swap (bool): Whether to swap left and right occlusion masks
        """
        if data_sample.get('occ_masks', None) is None:
            return

        occ_masks = list()
        for occ_mask in data_sample['occ_masks']:
            occ_masks.append(mmcv.imflip(occ_mask, self.direction))

        if swap:
            assert len(occ_masks) == 2
            occ_masks.reverse()

        data_sample['occ_masks'] = occ_masks

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Apply flip transform to stereo data sample.

        note:
            For horizontal flips:
            1. Verifies data sample is symmetric
            2. Flips all data
            3. Swaps left and right data to maintain positive disparity value
        keys:
        - optional(imgs)
        - optional(gt_disps)
        - optional(pred_disps)
        - optional(occ_masks)

        modify keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)
        add keys:
            - do_flip
            - flip_direction
        """
        if self.direction == 'horizontal':
            if not check_symmetric(results,
                                   check_keys=['imgs', 'gt_disps', 'pred_disps', 'occ_masks']):
                return results

        swap = self.direction == 'horizontal'
        self._flip_imgs(results, swap)
        self._flip_disps(results, swap, key='gt_disps')
        self._flip_disps(results, swap, key='pred_disps')
        self._flip_occ_masks(results, swap)
        return results


class Pad(BaseTransform):
    """Pad images and associated data with specified padding values.

    Args:
        padding (Union[int, Tuple[int, int], Tuple[int, int, int, int]]): Padding size.
            - If int, pad all borders with same value
            - If tuple of 2 ints, pad (top/bottom, left/right) with same values
            - If tuple of 4 ints, pad (top, bottom, left, right) with different values
        padding_mode (str): Padding mode, only 'edge' is supported for disparity maps.
            Defaults to 'edge'.
    """

    def __init__(self, padding, padding_mode='edge'):
        super().__init__()
        # todo: fix padding mode for disparity
        if padding_mode != 'edge':
            raise NotImplementedError("only support edge padding in stereo task, "
                                      "other padding mode may cause some problem and will fix in the future")
        self.padding = padding
        self.padding_mode = padding_mode

    def _pad_imgs(self, results: Dict):
        """Pad images in data sample."""
        if results.get('imgs', None) is None:
            return

        pad_width = parse_padding(self.padding)
        imgs = []
        for img in results['imgs']:
            imgs.append(np.pad(img, pad_width=pad_width, mode=self.padding_mode))
        results['imgs'] = imgs

        results['padding'] = self.padding
        results['padding_mode'] = self.padding_mode

    def _pad_disps(self, data_sample, key='gt_disps'):
        """Pad disparity maps in data sample."""
        if data_sample.get(key, None) is None:
            return

        disps = []
        for disp in data_sample[key]:
            assert hasattr(disp, 'pad')
            disps.append(disp.pad(self.padding))
        data_sample[key] = disps

    def _pad_occ_masks(self, data_sample):
        """Pad occlusion masks in data sample."""
        if data_sample.get('occ_masks', None) is None:
            return

        pad_width = parse_padding(self.padding)
        occ_masks = []
        for occ_mask in data_sample['occ_masks']:
            occ_masks.append(np.pad(occ_mask, pad_width=pad_width, mode=self.padding_mode))
        data_sample['occ_masks'] = occ_masks

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Apply padding transformation to data sample.
        keys:
        - optional(imgs)
        - optional(gt_disps)
        - optional(pred_disps)
        - optional(occ_masks)

        modify keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)
        add keys:
            - padding
            - padding_mode
        """
        self._pad_imgs(results)
        self._pad_disps(results, key='gt_disps')
        self._pad_disps(results, key='pred_disps')
        self._pad_occ_masks(results)
        return results


class PadToMultiple(BaseTransform):
    """Pad images and associated data to dimensions that are multiples of divisor.

    Args:
        divisor (int): The number that the padded dimensions should be divisible by.
            Defaults to 32.
        padding_mode (str): Padding mode, only 'edge' is supported for disparity maps.
            Defaults to 'edge'.
    """

    def __init__(self, divisor: int = 32, padding_mode='edge'):
        super().__init__()
        self.divisor = divisor
        self.padding_mode = padding_mode

    def transform(self, data_sample):
        """Apply padding to make dimensions multiples of divisor.
        keys:
        - optional(imgs)
        - optional(gt_disps)
        - optional(pred_disps)
        - optional(occ_masks)

        modify keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)
        add keys:
            - padding
            - padding_mode
        """
        # Get input image size
        if data_sample.get('imgs', None) is None:
            return data_sample

        h, w = data_sample['imgs'][0].shape[:2]

        # Calculate padding needed
        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor

        # Convert to 4-tuple padding format (top, bottom, left, right)
        padding = (pad_h // 2, pad_h - pad_h // 2,
                   pad_w // 2, pad_w - pad_w // 2)

        # Apply padding using Pad transform
        pad_transform = Pad(padding=padding, padding_mode=self.padding_mode)
        return pad_transform(data_sample)


class UnPad(BaseTransform):
    """Remove padding from images and associated data.

    Args:
        padding (Union[int, Tuple[int, int], Tuple[int, int, int, int]]): Padding to remove.
            - If int, remove same amount from all borders
            - If tuple of 2 ints, remove (top/bottom, left/right) padding
            - If tuple of 4 ints, remove (top, bottom, left, right) padding
    """

    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def _unpad_imgs(self, results):
        """Remove padding from images in data sample."""
        if results.get('imgs', None) is None:
            return

        pad_width = parse_padding(self.padding)
        imgs = []
        for img in results['imgs']:
            imgs.append(im_unpad(img, pad_width))
        results['imgs'] = imgs
        results['un_padding'] = self.padding

    def _unpad_disps(self, results, key='gt_disps'):
        """Remove padding from disparity maps in data sample."""
        if results.get(key, None) is None:
            return

        disps = []
        for disp in results[key]:
            assert hasattr(disp, 'unpad')
            disps.append(disp.unpad(self.padding))
        results[key] = disps

    def _unpad_occ_masks(self, results):
        """Remove padding from occlusion masks in data sample."""
        if results.get('occ_masks', None) is None:
            return

        pad_width = parse_padding(self.padding)
        occ_masks = []
        for occ_mask in results['occ_masks']:
            occ_masks.append(im_unpad(occ_mask, pad_width))
        results['occ_masks'] = occ_masks

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Apply unpadding transformation to data sample.
        keys:
        - optional(imgs)
        - optional(gt_disps)
        - optional(pred_disps)
        - optional(occ_masks)

        modify keys:
            - optional(imgs)
            - optional(gt_disps)
            - optional(pred_disps)
            - optional(occ_masks)
        add keys:
            - un_padding
        """
        self._unpad_imgs(results)
        self._unpad_disps(results, key='gt_disps')
        self._unpad_disps(results, key='pred_disps')
        self._unpad_occ_masks(results)
        return results


class RandomOcclusion(BaseTransform):
    """Randomly apply occlusions to the right image in stereo pairs.

    This transform:
    1. Adds random occlusions to right image only, using mean value of the occluded region
    2. Updates occlusion masks to mark occluded pixels (if exists)
    3. Uses right disparity map to find corresponding occluded points in left image

    Note:
        When occlusion masks exist, requires symmetric data
    """

    def __init__(self, n_holes: tuple = (1, 4),
                 size_range: tuple = ((20, 20), (60, 60))):
        """

        Args:
            n_holes: (min_holes, max_holes)
            size_range: ((min_h, min_w), (max_h, max_w))
        """
        super().__init__()
        assert len(n_holes) == 2 and n_holes[0] <= n_holes[1]
        assert len(size_range) == 2

        self.n_holes = n_holes
        self.size_range = size_range

    @cache_randomness
    def _generate_random_holes(self, img_h: int, img_w: int) -> List[Tuple[slice, slice]]:
        """Generate random occlusion regions."""
        n = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        holes = []

        for _ in range(n):
            # Random size
            h = np.random.randint(self.size_range[0][0], self.size_range[1][0] + 1)
            w = np.random.randint(self.size_range[0][1], self.size_range[1][1] + 1)

            # Random position
            top = np.random.randint(0, img_h - h + 1)
            left = np.random.randint(0, img_w - w + 1)

            holes.append((
                slice(top, top + h),
                slice(left, left + w)
            ))

        return holes

    @staticmethod
    def _apply_occlusion(
            results: Dict,
            holes: List[Tuple[slice, slice]]
    ) -> Dict:
        """Apply occlusions and update corresponding masks.

        Args:
            results (Dict): Data dictionary containing images and masks
            holes (List[Tuple[slice, slice]]): List of hole regions defined by slices

        Returns:
            Dict: Updated results with occlusions applied

        Notes:
            - Fills holes in right image with mean values
            - Updates occlusion masks if present
            - Handles both single and multi-channel images
        """
        # Early validation of required data
        if 'imgs' not in results or len(results['imgs']) != 2:
            return results

        img_right = results['imgs'][1]

        for row_slice, col_slice in holes:
            # Get and process hole region
            hole_region = img_right[row_slice, col_slice]

            # Calculate fill values based on image type
            if hole_region.ndim == 3:  # Multi-channel
                mean_values = np.mean(hole_region, axis=(0, 1), keepdims=True)
                hole_fill = np.broadcast_to(mean_values, hole_region.shape)
            else:  # Single channel
                mean_value = np.mean(hole_region)
                hole_fill = np.full_like(hole_region, mean_value)

            # Apply occlusion
            results['imgs'][1][row_slice, col_slice] = hole_fill

            # Update occlusion masks if they exist
            if 'occ_masks' in results and results['occ_masks'] is not None:
                img_h, img_w = results['imgs'][0].shape[:2]

                # Update right mask
                results['occ_masks'][1][row_slice, col_slice] = False

                # Update left mask if right disparity exists
                if ('gt_disps' in results and len(results['gt_disps']) > 1
                        and results['gt_disps'][1] is not None):
                    disp_right = results['gt_disps'][1]
                    y_coords, x_coords = np.mgrid[row_slice, col_slice]
                    valid = (x_coords >= 0) & (x_coords < img_w)

                    # Process valid pixels
                    disp_values = disp_right[row_slice, col_slice][valid]
                    y_left = y_coords[valid]
                    x_left = x_coords[valid] - disp_values.astype(np.int32)

                    # Update left mask for valid coordinates
                    valid_left = (x_left >= 0) & (x_left < img_w)
                    results['occ_masks'][0][y_left[valid_left], x_left[valid_left]] = False

        return results

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        """Apply random occlusions to stereo data."""

        # Check if masks exist and data is symmetric
        if results.get('occ_masks', None) is not None:
            if not check_symmetric(results, check_keys=['imgs', 'gt_disps', 'pred_disps', 'occ_masks']):
                return results

        # Generate random occlusion regions
        holes = self._generate_random_holes(results['imgs'][0].shape[:2])

        # Apply occlusions
        return self._apply_occlusion(results, holes)


@TRANSFORMS.register_module()
class RandomRotateShiftRight(BaseTransform):
    """Simulates imperfect camera calibration in stereo vision by applying random perturbations.

    This transform randomly rotates and shifts the right image to mimic real-world
    calibration errors that may occur due to mechanical vibrations, temperature changes,
    or installation imperfections.

    Args:
        angle (float): Maximum rotation angle in degrees. Default: 0.1
        pixel (int): Maximum horizontal shift in pixels. Default: 2
        p (float): Probability of applying the transform. Default: 0.5
    """

    def __init__(self, angle: float = 0.1, pixel: int = 2, p: float = 0.5):
        super().__init__()
        self.angle = angle
        self.pixel = pixel
        self.p = p

    def transform(self, results: Dict) -> Dict:
        """Apply random rotation and shift to the right image.

        Args:
            results (dict): Result dict containing the right image.
                Must have key 'img_right'.

        Returns:
            dict: Updated result dict.
        """
        if np.random.random() >= self.p:
            return results

        assert isinstance(results['imgs'], list)
        assert len(results['imgs']) == 2

        right_img = results['imgs'][1]

        # Generate random transformation parameters
        px = np.random.uniform(-self.pixel, self.pixel)
        ag = np.random.uniform(-self.angle, self.angle)

        # Random center point for rotation
        image_center = (
            np.random.uniform(0, right_img.shape[0]),
            np.random.uniform(0, right_img.shape[1])
        )

        # Apply rotation
        rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
        right_img = cv2.warpAffine(
            right_img,
            rot_mat,
            right_img.shape[1::-1],
            flags=cv2.INTER_LINEAR
        )

        # Apply translation
        trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
        right_img = cv2.warpAffine(
            right_img,
            trans_mat,
            right_img.shape[1::-1],
            flags=cv2.INTER_LINEAR
        )

        results['imgs'][1] = right_img
        return results

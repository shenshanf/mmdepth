
from abc import ABC, abstractmethod
import math
from typing import Optional

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness

from mmdepth.registry import TRANSFORMS

_MAX_LEVEL = 10


def level_to_mag(level: Optional[int], min_mag: float,
                 max_mag: float) -> float:
    """Map from level to magnitude."""
    if level is None:
        return round(np.random.rand() * (max_mag - min_mag) + min_mag, 1)
    else:
        return round(level / _MAX_LEVEL * (max_mag - min_mag) + min_mag, 1)


@TRANSFORMS.register_module()
class ColorTransform(BaseTransform, ABC):
    """Base class for color transformations. All color transformations need to
    inherit from this base class.

    Required Keys:
    - imgs

    Modified Keys:
    - imgs

    Args:
        prob (float): The probability for performing the geometric
            transformation and should be in range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for color transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for color transformation.
            Defaults to 1.9.
        asymmetric_prob (float): The probability for applying asymmetric transformations to left/right images.
    """
    name: Optional[str] = None

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9,
                 asymmetric_prob: float = 1.0) -> None:
        assert 0 <= prob <= 1.0, f'The probability of the transformation ' \
                                 f'should be in range [0,1], got {prob}.'
        assert level is None or isinstance(level, int), \
            f'The level should be None or type int, got {type(level)}.'
        assert level is None or 0 <= level <= _MAX_LEVEL, \
            f'The level should be in range [0,{_MAX_LEVEL}], got {level}.'
        assert isinstance(min_mag, float), \
            f'min_mag should be type float, got {type(min_mag)}.'
        assert isinstance(max_mag, float), \
            f'max_mag should be type float, got {type(max_mag)}.'
        assert min_mag <= max_mag, \
            f'min_mag should smaller than max_mag, ' \
            f'got min_mag={min_mag} and max_mag={max_mag}'

        self.prob = prob
        self.level = level
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.asymmetric_prob = asymmetric_prob

    @abstractmethod
    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Transform a single image.

        Args:
            img (np.ndarray): Input image.
            mag (float): Magnitude of the transformation.

        Returns:
            np.ndarray: Transformed image.
        """
        pass

    def _get_mag(self):
        """Get the magnitude of the transform.
        note: remove `cache_randomness` decorator
        """
        return level_to_mag(self.level, self.min_mag, self.max_mag)

    def _random_apply(self):
        """determine whether to apply transform
        note: remove `cache_randomness` decorator
        """
        return np.random.rand() <= self.prob

    @cache_randomness
    def _random_asymmetric(self):
        """determine whether asymmetric transform"""
        return np.random.rand() < self.asymmetric_prob

    def transform(self, results: dict) -> dict:
        """Transform function for images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """
        if results.get('imgs', None) is None:
            return results

        if self._random_asymmetric():
            # asymmetric
            imgs = []
            for img in results['imgs']:
                if np.random.rand() <= self.prob:
                    mag = self._get_mag()  # each image has own `mag`
                    img = self._transform_img(img, mag)
                imgs.append(img)
            results['imgs'] = imgs
            results[self.name] = 'asymmetric'
        else:
            # symmetric
            if np.random.rand() <= self.prob:  #
                mag = self._get_mag()  # share `mag`
                imgs = [self._transform_img(img, mag) for img in results['imgs']]
                results['imgs'] = imgs
                results[self.name] = 'symmetric'
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'level={self.level}, '
        repr_str += f'min_mag={self.min_mag}, '
        repr_str += f'max_mag={self.max_mag}, '
        repr_str += f'asymmetric_prob={self.asymmetric_prob})'
        return repr_str


@TRANSFORMS.register_module()
class Color(ColorTransform):
    """Adjust the color balance of the image, in a manner similar to the
    controls on a colour TV set. A magnitude=0 gives a black & white image,
    whereas magnitude=1 gives the original image. The bboxes, masks and
    segmentations are not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Color transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Color transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Color transformation.
            Defaults to 1.9.
    """
    name = 'color'

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Color should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Color should be in range [0,2], got {max_mag}.'
        super().__init__(prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img, mag: float, backend='cv2'):
        """Apply Color transformation to image."""
        return mmcv.adjust_color(img, mag, backend=backend)


@TRANSFORMS.register_module()
class Brightness(ColorTransform):
    """Adjust the brightness of the image. A magnitude=0 gives a black image,
    whereas magnitude=1 gives the original image. The bboxes, masks and
    segmentations are not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Brightness transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Brightness transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Brightness transformation.
            Defaults to 1.9.
    """
    name = 'brightness'

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Brightness should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Brightness should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img, mag: float, backend='cv2'):
        """Apply Color transformation to image."""
        return mmcv.adjust_brightness(img, mag)


@TRANSFORMS.register_module()
class Contrast(ColorTransform):
    """Control the contrast of the image. A magnitude=0 gives a gray image,
    whereas magnitude=1 gives the original imageThe bboxes, masks and
    segmentations are not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Contrast transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Contrast transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Contrast transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Contrast should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Contrast should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img, mag: float):
        """Apply Color transformation to image."""
        return mmcv.adjust_contrast(img, mag)


@TRANSFORMS.register_module()
class Sharpness(ColorTransform):
    """Adjust images sharpness. A positive magnitude would enhance the
    sharpness and a negative magnitude would make the image blurry. A
    magnitude=0 gives the origin img.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Sharpness transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Sharpness transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Sharpness transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Sharpness should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Sharpness should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img, mag: float):
        return mmcv.adjust_sharpness(img, mag)


@TRANSFORMS.register_module()
class Solarize(ColorTransform):
    """Solarize images (Invert all pixels above a threshold value of
    magnitude.).

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Solarize transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Solarize transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for Solarize transformation.
            Defaults to 256.0.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 256.0) -> None:
        assert 0. <= min_mag <= 256.0, f'min_mag for Solarize should be ' \
                                       f'in range [0, 256], got {min_mag}.'
        assert 0. <= max_mag <= 256.0, f'max_mag for Solarize should be ' \
                                       f'in range [0, 256], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img, mag: float):
        return mmcv.solarize(img, int(mag))


@TRANSFORMS.register_module()
class SolarizeAdd(ColorTransform):
    """SolarizeAdd images. For each pixel in the image that is less than 128,
       add an additional amount to it decided by the magnitude.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing SolarizeAdd
            transformation. Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for SolarizeAdd transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for SolarizeAdd transformation.
            Defaults to 110.0.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 110.0) -> None:
        assert 0. <= min_mag <= 110.0, f'min_mag for SolarizeAdd should be ' \
                                       f'in range [0, 110], got {min_mag}.'
        assert 0. <= max_mag <= 110.0, f'max_mag for SolarizeAdd should be ' \
                                       f'in range [0, 110], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img, mag: float):
        img_solarized = np.where(img < 128, np.minimum(img + mag, 255), img)
        return img_solarized.astype(img.dtype)


@TRANSFORMS.register_module()
class Posterize(ColorTransform):
    """Posterize images (reduce the number of bits for each color channel).

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Posterize
            transformation. Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Posterize transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for Posterize transformation.
            Defaults to 4.0.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 4.0) -> None:
        assert 0. <= min_mag <= 8.0, f'min_mag for Posterize should be ' \
                                     f'in range [0, 8], got {min_mag}.'
        assert 0. <= max_mag <= 8.0, f'max_mag for Posterize should be ' \
                                     f'in range [0, 8], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img, mag: float):
        return mmcv.posterize(img, math.ceil(mag)).astype(img.dtype)


@TRANSFORMS.register_module()
class Equalize(ColorTransform):
    """Equalize the image histogram. The bboxes, masks and segmentations are
    not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Equalize transformation.
            Defaults to 1.0.
        level (int, optional): No use for Equalize transformation.
            Defaults to None.
        min_mag (float): No use for Equalize transformation. Defaults to 0.1.
        max_mag (float): No use for Equalize transformation. Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag=None,
                 max_mag=None) -> None:
        assert level is None
        assert min_mag is None
        assert max_mag is None
        super().__init__(prob, level=0, min_mag=0, max_mag=0)

    def _transform_img(self, img, mag: float):
        return mmcv.imequalize(img).astype(img.dtype)


@TRANSFORMS.register_module()
class AutoContrast(ColorTransform):
    """Auto adjust image contrast.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing AutoContrast should
             be in range [0, 1]. Defaults to 1.0.
        level (int, optional): No use for AutoContrast transformation.
            Defaults to None.
        min_mag (float): No use for AutoContrast transformation.
            Defaults to 0.1.
        max_mag (float): No use for AutoContrast transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag=None,
                 max_mag=None) -> None:
        assert level is None
        assert min_mag is None
        assert max_mag is None
        super().__init__(prob, level=0, min_mag=0, max_mag=0)

    def _transform_img(self, img, mag: float):
        """Auto adjust image contrast."""
        return mmcv.auto_contrast(img)


@TRANSFORMS.register_module()
class Invert(ColorTransform):
    """Invert images.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing invert therefore should
             be in range [0, 1]. Defaults to 1.0.
        level (int, optional): No use for Invert transformation.
            Defaults to None.
        min_mag (float): No use for Invert transformation. Defaults to 0.1.
        max_mag (float): No use for Invert transformation. Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag=None,
                 max_mag=None) -> None:
        assert level is None
        assert min_mag is None
        assert max_mag is None
        super().__init__(prob, level=0, min_mag=0, max_mag=0)

    def _transform_img(self, img, mag: float):
        """Invert the image."""
        return mmcv.iminvert(img)

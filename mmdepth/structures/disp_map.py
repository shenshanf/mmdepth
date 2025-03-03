from typing import Optional, Tuple, Union, List

import mmcv
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from mmengine import print_log
from mmdepth.utils import parse_padding, resize_sparse_map

from .masked_map import MaskedMap
from .base import numpy_only
from .multi_level_data import MultilevelData


class DispMap(MaskedMap):
    """Data structure for disparity map with valid mask.

    Inherits from MaskData and adds disparity-specific operations.
    """

    def __init__(self, map=None, v_mask=None, scale_factor: float = 1.0):
        """note: record scale_factor relative to original shape

        Args:
            scale_factor
        """
        super().__init__(metainfo=dict(scale_factor=scale_factor),
                         map=map, v_mask=v_mask)

    def _get_norm_values(self,
                         norm_max: Optional[float] = None,
                         norm_min: Optional[float] = None,
                         percentile: int = 98) -> Tuple[float, float]:
        """Get normalization values for disparity visualization.

        Args:
            norm_max: Override maximum value for normalization
            norm_min: Override minimum value for normalization
            percentile: Percentile for computing max value if not specified

        Returns:
            Tuple[float, float]: (norm_max, norm_min) values
        """
        assert 90 <= percentile <= 100
        # Apply scale factor if values are provided
        scale = self.get('scale_factor', 1.0)
        valid_disp = self.filter_valid()
        assert valid_disp is not None
        if norm_max is not None:
            norm_max = norm_max * scale  # note: normalize value should be scale because disp value is scaled
        else:
            norm_max = np.percentile(valid_disp, percentile)
            print_log(f"Computing norm_max using percentile({percentile}), norm_max:{norm_max}",
                      logger='current', level=30)
        if norm_min is not None:
            norm_min = norm_min * scale  # note: normalize value should be scale because disp value is scaled
        else:
            norm_min = valid_disp.min()
            print_log(f"Computing norm_min using data minimum, norm_min:{norm_min}",
                      logger='current', level=30)

        return norm_max, norm_min

    def _default_color_mapper(self, norm_max, norm_min, cmap: str = 'turbo') -> ScalarMappable:
        """Create default color mapper for disparity visualization.

        This method uses _get_norm_values to determine normalization range
        and creates a default color mapper with 'turbo' colormap.

        Returns:
            ScalarMappable: A mapper that converts disparity values to RGB colors
        """
        # Get normalization values using existing method
        norm_max, norm_min = self._get_norm_values(norm_max, norm_min)

        # Create normalizer
        normalizer = Normalize(vmin=norm_min, vmax=norm_max, clip=True)

        # Get colormap
        cmap = plt.get_cmap(cmap)
        cmap.set_bad((0, 0, 0))  # Set invalid regions to black

        return ScalarMappable(norm=normalizer, cmap=cmap)

    @numpy_only
    def to_psd_img(self, norm_max=None,
                   norm_min=None,
                   cmap='turbo',
                   color_mapper: Optional[ScalarMappable] = None,
                   is_color_bar: bool = False,
                   to_bgr=True) -> Optional[np.ndarray]:
        """Convert disparity map to pseudo-colored visualization.
        note:

        Args:
            norm_max:
            norm_min:
            cmap: color map
            color_mapper: Optional custom ScalarMappable for color mapping
            is_color_bar: Whether to return colorbar (not implemented)
            to_bgr: whether convert color channels

        Returns:
            np.ndarray or None: Colored disparity map with shape (H, W, 3) if successful,
                None if input map is None.

        Raises:
            NotImplementedError: If is_color_bar is True.
        """
        if self.map is None:
            return None

        # Handle all invalid case
        if self.is_blank:
            if is_color_bar:
                raise NotImplementedError("Colorbar not implemented for all invalid case")
            return np.zeros((*self.shape, 3), dtype=np.uint8)

        # Create default color mapper if none provided
        if color_mapper is None:
            color_mapper = self._default_color_mapper(norm_max, norm_min, cmap)

        # Convert data to float and mark invalid pixels as NaN
        d_map = self.map_clone(invalid_value=np.nan)

        # Apply color mapping
        colored_map = color_mapper.to_rgba(d_map)[:, :, :3]  # only take rgb

        if is_color_bar:
            raise NotImplementedError("Colorbar support not implemented yet")

        if to_bgr:
            colored_map = colored_map[:, :, ::-1]  # revert channels

        return colored_map

    def pad(self, padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]], **kwargs) -> 'DispMap':
        """
        Note:
            Edge padding is used to maintain density of disparity map, but it introduces
            inaccurate disparity values at the boundaries (the padded disparities on
            left/right sides are not geometrically precise).

        """
        pad_width = parse_padding(padding)

        if not kwargs.get('padding_mode', 'edge') == 'edge':
            raise NotImplementedError("only support padding mode 'edge' for maintain sparsity")

        return self._convert(
            self,
            apply_to=np.ndarray,
            func=lambda x, pw: np.pad(x, pad_width=pw, mode='edge'),  # cv2.CopyMakeBorder is not support bool array
            pw=pad_width
        )

    def resize(self, out_size: Tuple[int, int], **kwargs) -> 'DispMap':
        """Resize map and update scale factor.

        Args:
            out_size: (width, height) of target size
        """
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

        return self.new(metainfo=dict(scale_factor=self.get('scale_factor') * w_scale),
                        map=dmap * w_scale,  # note: disparity value should scale at width change
                        v_mask=v_mask)

    def show(self, wait_time=0, *, win_name=None):
        """Visualization. (use in debug)

        Args:
            win_name
            wait_time: Time to wait before closing window.
        """
        assert isinstance(self._map, np.ndarray), "must convert to np.ndarray to show"
        psd_color_map = self.to_psd_img()
        assert psd_color_map is not None
        win_name = self.__class__.__name__ if win_name is None else win_name
        mmcv.imshow(psd_color_map, win_name=win_name, wait_time=wait_time)


class MultiDispMap(MultilevelData[DispMap]):
    """A class for managing multi-level disparity maps.
    """


DispMapType = Union[DispMap, MultiDispMap]

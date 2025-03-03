from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm, Colormap

import mmcv
from mmdepth.utils import resize_sparse_map, parse_padding, intersect_masks

from .masked_map import MaskedMap
from .disp_map import DispMap


class ErrorMap(MaskedMap):
    """Data structure for error map with valid mask.

    This class provides functionality for:
    1. Computing error between predicted and ground truth disparity maps
    2. Visualizing error maps with custom color schemes
    3. Supporting both absolute and relative error metrics
    """

    # Class-level constants
    DEFAULT_ABS_THRESH = 3.0  # Default absolute error threshold
    DEFAULT_REL_THRESH = 0.05  # Default relative error threshold (5%)

    def __init__(self, map, v_mask, scale_factor: float = 1.0):
        """note: record scale_factor relative to original shape

        Args:
            scale_factor
        """
        super().__init__(metainfo=dict(scale_factor=scale_factor),
                         map=map, v_mask=v_mask)

    def _default_color_mapper(self) -> ScalarMappable:
        """Create default color mapper for error visualization.

        The color scheme is designed to highlight different error ranges:
        - Cool colors (blues) for low errors
        - Neutral colors (whites) for medium errors
        - Warm colors (yellows to reds) for high errors

        Returns:
            ScalarMappable: A mapper that converts error values to RGB colors
        """
        # Define error boundaries (divided by 3.0 for normalization)
        bounds = np.array([0, 0.1875, 0.375, 0.75, 1.5, 3, 6, 12, 24, 48, np.inf]) / self.DEFAULT_ABS_THRESH

        # Define corresponding colors (normalized to 0-1)
        colors = np.array([
            [49, 54, 149],  # Deep blue (very small error)
            [69, 117, 180],  # Blue
            [116, 173, 209],  # Light blue
            [171, 217, 233],  # Very light blue
            [224, 243, 248],  # Near white
            [254, 224, 144],  # Light yellow
            [253, 174, 97],  # Orange
            [244, 109, 67],  # Orange red
            [215, 48, 39],  # Red
            [165, 0, 38]  # Deep red (very large error)
        ]) / 255.0

        # Create colormap
        cmap = ListedColormap(colors)
        cmap.set_bad((0, 0, 0))  # Set invalid regions to black

        # Create boundary normalizer
        norm = BoundaryNorm(boundaries=bounds, ncolors=len(bounds) - 1)

        return ScalarMappable(norm=norm, cmap=cmap)

    def to_psd_img(self,
                   color_mapper: Optional[ScalarMappable] = None,
                   is_color_bar: bool = False) -> Optional[np.ndarray]:
        """Convert error map to pseudo-colored visualization.

        Args:
            color_mapper: Optional custom ScalarMappable for color mapping
            is_color_bar: Whether to return colorbar (not implemented)

        Returns:
            np.ndarray or None: Colored error map if successful, None if input is None
        """
        if self.map is None:
            return None

        # Use default or custom color mapper
        mapper = self._default_color_mapper() if color_mapper is None else color_mapper

        # Convert data type and handle invalid values
        e_map = self.map_clone(invalid_value=np.nan)

        # note: scale back the value relative to original shape
        e_map = e_map / self.get('scale_factor', 1.0)

        # Convert to RGB image using mapper
        colored_map = mapper.to_rgba(e_map)[:, :, :3]

        if is_color_bar:
            raise NotImplementedError("Colorbar support not implemented yet")

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

    def resize(self, out_size: Tuple[int, int], **kwargs) -> 'ErrorMap':
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

    @classmethod
    def kitti_d1(cls,
                 gt_disp: DispMap,
                 pr_disp: DispMap,
                 mask=None,
                 abs_thresh: float = DEFAULT_ABS_THRESH,
                 rel_thresh: float = DEFAULT_REL_THRESH) -> 'ErrorMap':
        """Compute error map between ground truth and predicted disparity maps.

        The error is computed as the minimum of:
        1. Absolute error normalized by abs_thresh
        2. Relative error normalized by rel_thresh

        Args:
            gt_disp: Ground truth disparity map
            pr_disp: Predicted disparity map
            mask:
            abs_thresh: Absolute error threshold (default: 3.0)
            rel_thresh: Relative error threshold (default: 0.05)

        Returns:
            ErrorMap: Computed error map

        Raises:
            TypeError: If inputs are not DispMap objects
            ValueError: If maps are None or shapes don't match
        """
        # Input validation
        if not isinstance(gt_disp, DispMap) or not isinstance(pr_disp, DispMap):
            raise TypeError('Both inputs must be DispMap objects')

        if gt_disp.map is None or pr_disp.map is None:
            raise ValueError('Cannot compute error when either map is None')

        if gt_disp.shape != pr_disp.shape:
            raise ValueError(f'Shape mismatch: {gt_disp.shape} != {pr_disp.shape}')

        if gt_disp.get('scale_factor') != pr_disp.get('scale_factor'):
            raise ValueError(f"scale_factor mismatch: "
                             f"{gt_disp.get('scale_factor')}!={pr_disp.get('scale_factor')}")

        # Determine valid region (intersection of both masks)
        valid_mask = intersect_masks(gt_disp.v_mask, pr_disp.v_mask, mask)

        # compute epe
        emap_data = np.zeros_like(gt_disp.map, dtype=np.float32)

        # Compute error in valid regions
        valid_gt = gt_disp.filter_valid(valid_mask)
        valid_pr = pr_disp.filter_valid(valid_mask)

        # note: the config thresh is relative to original shape
        abs_thresh *= gt_disp.get('scale_factor')
        rel_thresh *= gt_disp.get('scale_factor')

        # d1 error
        emap_data[valid_mask] = np.minimum(np.abs(valid_gt - valid_pr) / abs_thresh,
                                           np.abs(valid_gt - valid_pr) / (valid_gt + 1e-6) / rel_thresh)

        return cls(map=emap_data, v_mask=valid_mask,
                   scale_factor=gt_disp.get('scale_factor', 1.0))

    @classmethod
    def epe(cls,
            gt_disp: DispMap,
            pr_disp: DispMap,
            mask=None) -> 'ErrorMap':
        """Compute error map between ground truth and predicted disparity maps.

        Args:
            gt_disp: Ground truth disparity map
            pr_disp: Predicted disparity map
            mask:
        Returns:
            ErrorMap: Computed error map

        Raises:
            TypeError: If inputs are not DispMap objects
            ValueError: If maps are None or shapes don't match
        """
        # Input validation
        if not isinstance(gt_disp, DispMap) or not isinstance(pr_disp, DispMap):
            raise TypeError('Both inputs must be DispMap objects')

        if gt_disp.map is None or pr_disp.map is None:
            raise ValueError('Cannot compute error when either map is None')

        if gt_disp.shape != pr_disp.shape:
            raise ValueError(f'Shape mismatch: {gt_disp.shape} != {pr_disp.shape}')

        if gt_disp.get('scale_factor') != pr_disp.get('scale_factor'):
            raise ValueError(f"scale_factor mismatch: "
                             f"{gt_disp.get('scale_factor')}!={pr_disp.get('scale_factor')}")

        # Determine valid region (intersection of both masks)
        valid_mask = intersect_masks(gt_disp.v_mask, pr_disp.v_mask, mask)

        # compute epe
        emap_data = np.zeros_like(gt_disp.map, dtype=np.float32)

        # Compute error in valid regions
        valid_gt = gt_disp.filter_valid(valid_mask)
        valid_pr = pr_disp.filter_valid(valid_mask)

        # d1 error
        emap_data[valid_mask] = np.abs(valid_gt - valid_pr)

        return cls(map=emap_data, v_mask=valid_mask,
                   scale_factor=gt_disp.get('scale_factor', 1.0))

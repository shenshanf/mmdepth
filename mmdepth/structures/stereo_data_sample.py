from typing import List, Optional, Union, Tuple, Any

import torch
import numpy as np

from .base import BaseDataSample

from .disp_map import DispMap, MultiDispMap, DispMapType

MapType = Union[np.ndarray, torch.Tensor]


class StereoDataSample(BaseDataSample):
    """A data structure for managing stereo matching data samples.

    The StereoDataSample class provides a structured container for stereo vision data,
    including stereo image pairs, ground truth disparity maps, and prediction results.
    It organizes data into three main fields:

    Fields:
        asset:
            Data elements that represent input or ground truth:
            - imgs (List[MapType]): A list containing stereo image pair [left_img, right_img]
                - Each image can be either torch.Tensor[C,H,W] or numpy.ndarray[H,W,C]
                - Channel dimension C must be either 1 (grayscale) or 3 (RGB)
                - Both images must have the same spatial dimensions
            - gt_disps (List[DispMap]): Ground truth disparity maps [left_disp, right_disp]
                - Each element is a DispMap instance containing the disparity map and valid mask
                - Left disparity is mandatory, right is optional
                - Disparity values represent pixel offsets between stereo pairs
            Additional asset fields can be added using set_asset()

        result:
            Computed outputs and predictions:
            - pred_disp (DispMapType): Predicted disparity map
                - Can be a single DispMap for one prediction level
                - Can be a MultiDispMap for multi-scale predictions
            Additional result fields can be added using set_result()

        metainfo:
            Auxiliary information about the data sample
            - Stores metadata like camera parameters, scene info, etc.
            - Can be set and extended using set_metainfo()

    Field Management:
        The class provides three methods for adding new fields:
        - set_metainfo(): Add auxiliary information fields to metainfo
        - set_asset(): Add new data elements to the asset field
        - set_result(): Add new outputs to the result field

    The class also provides properties and methods to:
        - Access and modify individual components (left/right images, disparity maps)
        - Validate data consistency (e.g., image dimensions)
        - Check asset symmetry (whether both left and right data are available)
        - Handle both single-level and multi-level disparity predictions
    """

    # imgs
    @property
    def imgs(self) -> List[MapType]:
        """Get image list [left, right]."""
        return getattr(self, '_imgs', None)

    @staticmethod
    def _check_imgs(value):
        # Validate input is length-2 sequence
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Input must be a length-2 sequence (left, right)")
        value = list(value)

        # Get shapes and validate data
        shapes = []
        for i, img in enumerate(['left', 'right']):
            v = value[i]

            if not isinstance(v, (torch.Tensor, np.ndarray)):
                raise TypeError(f"{img} image must be tensor or array")

            # Handle different dimension orders
            if isinstance(v, torch.Tensor):
                if v.dim() != 3 or v.size(0) not in (1, 3):
                    raise ValueError(f"{img} tensor must have shape [1/3,H,W]")
                shapes.append(v.shape[1:])
            else:
                if v.ndim != 3 or v.shape[-1] not in (1, 3):
                    raise ValueError(f"{img} array must have shape [H,W,1/3]")
                shapes.append(v.shape[:2])

        # Validate shape consistency
        if shapes[0] != shapes[1]:
            raise ValueError(f"Shape mismatch: left={shapes[0]}, right={shapes[1]}")

    @imgs.setter
    def imgs(self, value: List[MapType]):
        """Set image list [left, right].
        """
        if value is None:
            return
        self._check_imgs(value)
        self.set_field(value, '_imgs', dtype=None, field_type='asset')

    @imgs.deleter
    def imgs(self):
        """delete imgs"""
        if hasattr(self, '_imgs'):
            del self._imgs

    @property
    def img_left(self) -> Optional[MapType]:
        """Get left image."""
        if self.imgs is None:
            return None
        return self.imgs[0] if len(self.imgs) > 0 else None

    @property
    def img_right(self) -> Optional[MapType]:
        """Get left image."""
        if self.imgs is None:
            return None
        return self.imgs[1] if len(self.imgs) > 1 else None

    @property
    def gt_disps(self) -> List[DispMap]:
        """Get ground truth disparity maps [left, right]."""
        return getattr(self, '_gt_disps', None)

    @gt_disps.setter
    def gt_disps(self, value):
        """Set ground truth disparity maps [left, right].

        Args:
            value: Ground truth disparity data in one of formats:
                - map: Set left disparity only, no mask
                - Tuple[map, mask]: Set left disparity with optional mask
                - List[map/Tuple[map, mask], ...]: Set both left and right
                where map is tensor/array and mask is optional tensor/array
        """
        disp_maps = []

        # left: (map, v_mask=None) or(map, v_mask)
        if isinstance(value, (torch.Tensor, np.ndarray, tuple)):
            disp_maps.append(DispMap.from_tuple(value))

        # left and right
        elif isinstance(value, list):
            if len(value) > 2:
                raise ValueError("Input list must contain 1 or 2 elements:(left and right)")

            for v in value:
                if isinstance(v, (torch.Tensor, np.ndarray, tuple)):
                    disp_maps.append(DispMap.from_tuple(v))
        else:
            raise TypeError("Input must be tensor, array, tuple or list")

        self.set_field(disp_maps, '_gt_disps', dtype=list)

    @gt_disps.deleter
    def gt_disps(self):
        if hasattr(self, '_gt_disps'):
            del self._gt_disps

    @property
    def gt_disp(self) -> Optional[DispMap]:
        """Get left ground truth disparity map."""
        if self.gt_disps is None:
            return None
        return self.gt_disps[0] if len(self.gt_disps) > 0 else None

    #
    @property
    def gt_disp_right(self) -> Optional[DispMap]:
        """Get left ground truth disparity map."""
        if self.gt_disps is None:
            return None
        return self.gt_disps[1] if len(self.gt_disps) > 1 else None

    @property
    def pred_disp(self) -> Optional[DispMapType]:
        return getattr(self, '_pred_disp', None)

    @pred_disp.setter
    def pred_disp(self, value):
        if isinstance(value, (torch.Tensor, np.ndarray, tuple)):
            disp = DispMap.from_tuple(value)
        elif isinstance(value, list):
            # multi level: [[(d,v),(d,v),...], ...] or [[d,d,d,d,...],...]
            disp = MultiDispMap(data=[DispMap.from_tuple(obj) for obj in value])
        else:
            raise TypeError
        self.set_field(disp, '_pred_disp', field_type='result')

    @pred_disp.deleter
    def pred_disp(self):
        if hasattr(self, '_pred_disp'):
            del self._pred_disp

    def __getitem__(self, key: str) -> Any:
        """Get a field value from the data sample."""
        if key not in self.all_keys():
            raise KeyError(f'Key {key} not found in data sample')

        return self.get(key)

    @property
    def symmetric_asset(self) -> bool:
        """Check if the asset is symmetric.
        """
        for asset in self.asset_items():
            # Unpack the value
            _, value = asset

            # Check if value is a list and meets symmetry criteria
            if not isinstance(value, list):
                return False

            if len(value) != 2:
                return False

            if not all(item is not None for item in value):
                return False

        return True

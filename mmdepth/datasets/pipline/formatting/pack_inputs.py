from typing import Dict, Optional, Tuple, List, Union
import torch

import mmcv
from mmcv.transforms import BaseTransform

from mmdepth.registry import TRANSFORMS
from mmdepth.structures import StereoDataSample


@TRANSFORMS.register_module()
class PackStereoInputs(BaseTransform):
    """Pack the inputs data for stereo model.

    This transform packs stereo data into a standardized format:
    - Stacks input image pairs along specified dimension
    - Creates a StereoDataSample with data fields and metainfo

    Args:
        img_stack_dim (int): Dimension to stack stereo image pair.
            0 for batch dimension, 1 for channel dimension, None means no stack
            Defaults to None.
    """

    def __init__(self, img_stack_dim: Optional[int] = None,
                 data_keys=['imgs', 'gt_disps', 'pred_disps', 'occ_masks'],
                 metainfo_keys: List[str] = ['img_shape', 'ori_shape', 'file_paths', 'sample_idx']) -> None:
        if img_stack_dim not in [0, 1, None]:
            raise ValueError("image pair could only stack at batch or channel dim or not stack")
        self.img_stack_dim = img_stack_dim
        self.data_keys = data_keys
        self.metainfo_keys = metainfo_keys

    def transform(self, results: Dict) -> Dict:
        """Transform stereo inputs data.

        Args:
            results (dict): Dictionary containing:
                - imgs (List[torch.Tensor]): Left and right images
                - gt_disps (optional, List): Ground truth disparity maps
                - pred_disps (optional, List): Predicted disparity maps
                - occ_masks (optional, List): Occlusion masks
                - Other metainfo like img_shape, ori_shape, etc.

        Returns:
            dict: A dictionary containing:
                - inputs (torch.Tensor): Stacked and normalized stereo images
                - data_sample (StereoDataSample): Processed data sample
        """
        if 'imgs' not in results:
            raise KeyError("'imgs' not found in input results")

        packed_results = dict()
        # Stack image pair along specified dimension
        if self.img_stack_dim is not None:
            # note: if stack at batch dimension, it will collate as [bs*2, c, h, w]
            #       use: 'input.reshape(bs,2,c,h,w).chunk(dim=1)' to unpack left and right
            packed_results['inputs'] = torch.stack(results['imgs'], dim=self.img_stack_dim)
        else:
            # will collate as list([b,c,h,w], [b,c,h,w])
            packed_results['inputs'] = results['imgs']

            # Create data sample
        data_sample = StereoDataSample()

        # Pack data fields and metainfo
        for key in self.data_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='asset')

        for key in self.metainfo_keys:
            if key in results:
                data_sample.set_field(results[key], key, field_type='metainfo')
            else:
                raise KeyError(f"metainfo key: {key} not in results")

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        """Get string representation."""
        repr_str = self.__class__.__name__
        repr_str += f'(img_stack_dim={self.img_stack_dim})'
        return repr_str

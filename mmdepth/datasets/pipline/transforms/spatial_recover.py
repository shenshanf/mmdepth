from typing import Dict, List
from mmcv.transforms import BaseTransform
from mmdepth.registry import TRANSFORMS
from .spatial_transforms import Resize, UnPad, Flip


@TRANSFORMS.register_module()
class RecoverSpatialTransform(BaseTransform):
    """Transform to recover previous spatial transformations.

    This transform reverses the effects of resize, pad and flip operations
    in reverse order of their application. Note that crop operation cannot
    be recovered due to information loss.

    Args:
        transform_order (List[str]): Order of original transformations.
            Must be subset of ['resize', 'crop', 'pad', 'flip'].
            Defaults to ['resize', 'crop', 'pad', 'flip'].
    """

    def __init__(self, transform_order: List[str] = ['resize', 'crop', 'pad', 'flip']):
        super().__init__()
        valid_transforms = {'resize', 'crop', 'pad', 'flip'}
        if not all(t in valid_transforms for t in transform_order):
            raise ValueError(f'Transform order must be subset of {valid_transforms}')
        self.transform_order = transform_order

    @staticmethod
    def _recover_flip(results: Dict) -> Dict:
        """Recover flip transformation if applied."""
        if results.get('do_flip', False):
            flip_transform = Flip(direction=results['flip_direction'])
            results = flip_transform(results)
            # Clear flip flags
            results.pop('do_flip')
            results.pop('flip_direction')
        return results

    @staticmethod
    def _recover_pad(results: Dict) -> Dict:
        """Recover padding if applied."""
        if 'padding' in results:
            unpad_transform = UnPad(padding=results['padding'])
            results = unpad_transform(results)
            # Clear padding info
            results.pop('padding')
            results.pop('padding_mode')
        return results

    @staticmethod
    def _recover_resize(self, results: Dict) -> Dict:
        """Recover resize transformation if applied.

        The target size depends on whether crop was applied:
        - If cropped: recover to crop_size
        - If not cropped: recover to ori_shape
        """
        if 'scale_factor' in results:
            # Get target size based on whether crop was applied
            if 'crop_size' in results:
                target_h, target_w = results['crop_size']
            else:
                target_h, target_w = results['ori_shape']

            # Apply reverse resize
            resize_transform = Resize(
                scale=(target_w, target_h),
                interpolation='bilinear'
            )
            results = resize_transform(results)

            # Clear resize info
            results.pop('scale_factor')
            results.pop('keep_ratio', None)
            results.pop('scale', None)

        return results

    def transform(self, results: Dict) -> Dict:
        """Apply recovery transformations in reverse order.

        Args:
            results (Dict): Result dict containing transformation flags.

        Returns:
            Dict: Results after recovery transformations.
        """
        # Process transforms in reverse order
        recover_order = self.transform_order[::-1]

        for t in recover_order:
            if t == 'flip':
                results = self._recover_flip(results)
            elif t == 'pad':
                results = self._recover_pad(results)
            elif t == 'resize':
                results = self._recover_resize(results)
            elif t == 'crop':
                pass  # note: Crop cannot be recovered, skip it
            else:
                raise NotImplemented

        return results

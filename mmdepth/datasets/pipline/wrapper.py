from typing import List, Dict, Callable, Optional, Union
import numpy as np
from random import shuffle
from mmcv.transforms import KeyMapper, Compose
from mmcv.transforms.utils import cache_randomness
from mmdepth.registry import TRANSFORMS
from multimethod import multimethod


@TRANSFORMS.register_module()
class RandomShuffleCompose(Compose):
    """Compose multiple transforms with random shuffle.

    This class enables random shuffling of transforms while preserving group dependencies.
    Transforms within the same group maintain their relative order during shuffling.

    Args:
        transforms (list[dict | callable]): Sequence of transform configs or
            callable objects to be composed.
        groups (list[list[int]], optional): Groups of transform indices that
            should maintain their relative order. Each inner list represents
            a group of transforms that will stay together. Default: None
        prob (float): Probability of applying the shuffle. Default: 1.0

    Example:
        >>> # Simple random shuffle without groups
        >>> pipeline = dict(
        >>>     type='RandomShuffleCompose',
        >>>     transforms=[
        >>>         dict(type='Resize'),
        >>>         dict(type='RandomFlip'),
        >>>         dict(type='Normalize'),
        >>>     ])
        >>>
        >>> # With grouped transforms
        >>> pipeline = dict(
        >>>     type='RandomShuffleCompose',
        >>>     transforms=[
        >>>         dict(type='LoadImageFromFile'),  # group 1
        >>>         dict(type='LoadAnnotations'),    # group 1
        >>>         dict(type='Resize'),            # group 2
        >>>         dict(type='RandomFlip'),        # group 2
        >>>         dict(type='Normalize'),         # group 3
        >>>     ],
        >>>     groups=[[0,1], [2,3], [4]])        # define groups
    """

    def __init__(self,
                 transforms: List[Union[Dict, Callable]],
                 groups: Optional[List[List[int]]] = None,
                 prob: float = 1.0):
        super().__init__(transforms=transforms)
        self.groups = groups
        self.prob = prob

        # Validate group configuration if provided
        if self.groups is not None:
            self._validate_groups()

    def _validate_groups(self):
        """Validate that group indices are valid and non-overlapping."""
        # Track all indices used in groups
        all_indices = set()

        for group in self.groups:
            for idx in group:
                # Check index bounds
                if idx < 0 or idx >= len(self.transforms):
                    raise ValueError(
                        f'Transform index {idx} out of range '
                        f'[0, {len(self.transforms) - 1}]')

                # Check for duplicate indices
                if idx in all_indices:
                    raise ValueError(f'Duplicate transform index {idx} in groups')
                all_indices.add(idx)

    @cache_randomness
    def _random_shuffle(self) -> bool:
        """Determine whether to apply shuffling based on probability."""
        return np.random.rand() < self.prob

    @cache_randomness
    def _get_shuffle_indices(self) -> List[int]:
        """Generate shuffled indices while respecting group constraints.

        For grouped transforms, treats each group as a unit during shuffling.
        Ungrouped transforms are treated as single-element groups.

        Returns:
            list[int]: A shuffled list of transform indices.
        """
        n_transforms = len(self.transforms)

        # Simple case: no groups defined, shuffle all indices
        if self.groups is None:
            indices = list(range(n_transforms))
            np.random.shuffle(indices)
            return indices

        # Complex case: maintain group relationships
        # 1. Identify indices not in any group
        grouped_indices = set(idx for group in self.groups for idx in group)
        ungrouped_indices = list(set(range(n_transforms)) - grouped_indices)

        # 2. Create shuffling units (groups and individual transforms)
        units = []
        # Add transform groups
        for group in self.groups:
            units.append(group)
        # Add ungrouped transforms as single-element units
        for idx in ungrouped_indices:
            units.append([idx])

        # 3. Shuffle the units  note: fix?
        np.random.shuffle(units)
        # shuffle(units)

        # 4. Flatten units back to a single index list
        shuffled_indices = []
        for unit in units:
            shuffled_indices.extend(unit)

        return shuffled_indices

    def transform(self, results: Dict) -> Optional[Dict]:
        """Apply transforms to the data with optional shuffling.

        Args:
            results (dict): Data to be transformed.

        Returns:
            dict | None: Transformed data or None if any transform returns None.
        """
        # Check if shuffling should be applied
        if not self._random_shuffle():
            # No shuffle, use parent class implementation
            return super().transform(results)

        # Get shuffled ordering of transforms
        indices = self._get_shuffle_indices()

        # Apply transforms in shuffled order
        for idx in indices:
            results = self.transforms[idx](results)
            if results is None:
                return None

        # todo: 记录顺序
        return results

    def __repr__(self):
        """Generate string representation of the compose."""
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'    transforms={self.transforms}\n'
        repr_str += f'    groups={self.groups}\n'
        repr_str += f'    prob={self.prob}\n'
        repr_str += ')'
        return repr_str


@TRANSFORMS.register_module()
class StereoKeyMapper(KeyMapper):
    """A KeyMapper for stereo data that handles list-format data with internal state management.

    Maps list-format stereo data (e.g. imgs, disps) to standard keys for processing,
    using an internal state dictionary to avoid modifying input directly.
    """

    def __init__(self,
                 transforms: List[Union[Dict, Callable]],
                 mapping: Optional[Dict] = None,
                 remapping: Optional[Dict] = None,
                 auto_remap: Optional[bool] = None,
                 allow_nonexist_keys: bool = True):
        super().__init__(
            transforms=transforms,
            mapping=mapping,
            remapping=remapping,
            auto_remap=auto_remap,
            allow_nonexist_keys=allow_nonexist_keys)
        # 内部状态字典
        self._intermediate_data = {}

    def _reset_state(self):
        """Reset internal state dictionary."""
        self._intermediate_data.clear()

    def _map_input(self, data: Dict, mapping: Optional[Dict]) -> None:
        """Map inputs to internal state dictionary."""
        if mapping is None:
            return

        for inner_key, outer_key in mapping.items():
            if isinstance(outer_key, str):
                # 单列表映射 (e.g. 'imgs' -> 'img')
                if outer_key in data and isinstance(data[outer_key], list):
                    for idx, item in enumerate(data[outer_key]):
                        self._intermediate_data[f'{inner_key}_{idx}'] = item
            elif isinstance(outer_key, (list, tuple)):
                # 多列表映射 (e.g. ['gt_disps', 'pred_disps'] -> 'disp')
                for list_key in outer_key:
                    if list_key in data and isinstance(data[list_key], list):
                        for idx, item in enumerate(data[list_key]):
                            self._intermediate_data[f'{inner_key}_{list_key}_{idx}'] = item

    @multimethod
    def _apply_transforms(self):
        """Apply transforms to internal state data."""
        try:
            for t in self.transforms:

                transform_input = {}


                for key in self._intermediate_data:

                    base_key = key.split('_')[0]
                    transform_input[base_key] = self._intermediate_data[key]


                transform_output = t(transform_input)
                if transform_output is None:
                    return False


                for key, value in transform_output.items():
                    self._intermediate_data[key] = value
        except Exception as e:
            print(f"Error in transform: {str(e)}")

    def _map_output(self, data: Dict, remapping: Optional[Dict]) -> None:
        """Map internal state back to output format."""
        if remapping is None:
            return


        reconstructed = {}

        for inner_key, outer_key in remapping.items():
            if isinstance(outer_key, str):
                prefix = f'{inner_key}_'
                indices = sorted([
                    int(k.split('_')[-1]) for k in self._intermediate_data.keys()
                    if k.startswith(prefix) and k.split('_')[-1].isdigit()
                ])

                if indices:
                    items = [self._intermediate_data[f'{prefix}{i}'] for i in indices]
                    reconstructed[outer_key] = items

            elif isinstance(outer_key, (list, tuple)):

                for list_key in outer_key:

                    prefix = f'{inner_key}_{list_key}_'
                    indices = sorted([
                        int(k.split('_')[-1]) for k in self._intermediate_data.keys()
                        if k.startswith(prefix) and k.split('_')[-1].isdigit()
                    ])

                    if indices:
                        items = [self._intermediate_data[f'{prefix}{i}'] for i in indices]
                        reconstructed[list_key] = items


        data.update(reconstructed)

    def transform(self, results: Dict) -> Dict:
        """Apply the full transform pipeline using internal state management."""
        try:

            self._reset_state()


            self._map_input(results, self.mapping)


            self._apply_transforms()


            self._map_output(results, self.remapping or self.mapping)

            return results

        finally:
            self._reset_state()




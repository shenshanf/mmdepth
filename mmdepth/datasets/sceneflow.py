from typing import List, Union, Optional, Sequence, Callable

from natsort import natsorted

from mmengine.fileio import join_path, exists, list_from_file
from mmdepth.registry import DATASETS
from mmdepth.utils import natsort_iglob, equal_nonempty_length
from .base import BaseStereoDataset


@DATASETS.register_module()
class Flyingthings3d(BaseStereoDataset):
    """FlyingThings3D dataset for stereo matching.

    A synthetic dataset containing rendered scenes with ground truth disparity.
    For detailed information visit:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    """

    METAINFO = {
        'data_set_name': 'flyingthings3d',
        'task_type': 'standard_stereo_matching',
        'dataset_type': 'synthetic',
        'disp_type': 'dense',
        'has_occlusion': True,
        'disp_max': 192,
        'disp_min': 0,
        'ori_height': 540,
        'ori_width': 960,
        'license': 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License',
        'version': '1.0',
        'url': 'https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html'
    }

    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_root: Optional[str] = '',
                 passes: str = 'frame_cleanpass',
                 splits: str = 'TRAIN',
                 load_right_label: bool = False,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        """Initialize FlyingThings3D dataset.

        Args:
            ann_file: Path to annotation file.
            data_root: Root path for dataset.
            passes: Data pass type, options: ['frame_cleanpass', 'frame_finalpass', '*'].
            splits: Dataset split, options: ['TRAIN', 'TEST', '*'].
            load_right_label: Whether to load right view labels.
            indices: Indices to subset the dataset.
            serialize_data: Whether to serialize dataset.
            pipeline: Transform pipeline.
            test_mode: Whether in test mode.
            lazy_init: Whether to use lazy initialization.
            max_refetch: Maximum refetch count.
        """
        valid_passes = ['frame_cleanpass', 'frame_finalpass', '*']
        valid_splits = ['TRAIN', 'TEST', '*']

        if passes not in valid_passes:
            raise ValueError(f'passes must be one of {valid_passes}')
        if splits not in valid_splits:
            raise ValueError(f'splits must be one of {valid_splits}')

        data_prefix = dict(
            img_left=join_path('FlyingThings3D', passes, splits),
            img_right=join_path('FlyingThings3D', passes, splits),
            gt_disp_left=join_path('FlyingThings3D', 'disparity', splits),
            gt_disp_right=join_path('FlyingThings3D', 'disparity', splits),
        )

        super().__init__(
            ann_file=ann_file,
            metainfo=None,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=None,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            load_right_label=load_right_label,
            lazy_init=lazy_init,
            max_refetch=max_refetch
        )

    def glob_data_list(self) -> List[dict]:
        """Glob and organize stereo image pairs and disparity data.

        Returns:
            List[dict]: List of data samples with image and disparity paths.
        """
        img_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'img_{side}'],
            f'*/*/{side}/*.png'
        )
        disp_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'gt_disp_{side}'],
            f'*/*/{side}/*.pfm'
        )

        left_imgs = natsort_iglob(img_pattern('left'))
        right_imgs = natsort_iglob(img_pattern('right'))
        left_disps = natsort_iglob(disp_pattern('left'))

        if self.load_right_label:
            right_disps = natsort_iglob(disp_pattern('right'))
            assert equal_nonempty_length(left_imgs, right_imgs, left_disps, right_disps)
            return [
                dict(
                    imgs=[l, r],
                    gt_disps=[dl, dr]
                )
                for l, r, dl, dr in zip(left_imgs, right_imgs, left_disps, right_disps)
            ]

        assert equal_nonempty_length(left_imgs, right_imgs, left_disps)
        return [
            dict(
                imgs=[l, r],
                gt_disps=[dl]
            )
            for l, r, dl in zip(left_imgs, right_imgs, left_disps)
        ]


@DATASETS.register_module()
class Monkaa(BaseStereoDataset):
    """Monkaa dataset for stereo matching.

    A synthetic dataset from the SceneFlow dataset collection containing rendered scenes
    with ground truth disparity.
    """

    METAINFO = {
        'data_set_name': 'monkaa',
        'task_type': 'standard_stereo_matching',
        'dataset_type': 'synthetic',
        'disp_type': 'dense',
        'has_occlusion': True,
        'disp_max': 192,
        'disp_min': 0,
        'ori_height': 540,
        'ori_width': 960,
        'license': 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License',
        'version': '1.0',
        'url': 'https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html'
    }

    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_root: Optional[str] = '',
                 passes: str = 'frame_cleanpass',
                 load_right_label: bool = False,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        """Initialize Monkaa dataset.

        Args:
            ann_file: Path to annotation file.
            data_root: Root path for dataset.
            passes: Data pass type, options: ['frame_cleanpass', 'frame_finalpass'].
            load_right_label: Whether to load right view labels.
            indices: Indices to subset the dataset.
            serialize_data: Whether to serialize dataset.
            pipeline: Transform pipeline.
            test_mode: Whether in test mode.
            lazy_init: Whether to use lazy initialization.
            max_refetch: Maximum refetch count.
        """
        valid_passes = ['frame_cleanpass', 'frame_finalpass', '*']

        if passes not in valid_passes:
            raise ValueError(f'passes must be one of {valid_passes}')

        data_prefix = dict(
            img_left=join_path('Monkaa', passes),
            img_right=join_path('Monkaa', passes),
            gt_disp_left=join_path('Monkaa', 'disparity'),
            gt_disp_right=join_path('Monkaa', 'disparity'),
        )

        super().__init__(
            ann_file=ann_file,
            metainfo=None,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=None,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            load_right_label=load_right_label,
            lazy_init=lazy_init,
            max_refetch=max_refetch
        )

    def glob_data_list(self) -> List[dict]:
        """Glob and organize stereo image pairs and disparity data.

        Returns:
            List[dict]: List of data samples with image and disparity paths.
        """
        img_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'img_{side}'],
            f'*/{side}/*.png'  # Monkaa specific pattern
        )
        disp_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'gt_disp_{side}'],
            f'*/{side}/*.pfm'  # Monkaa specific pattern
        )

        left_imgs = natsort_iglob(img_pattern('left'))
        right_imgs = natsort_iglob(img_pattern('right'))
        left_disps = natsort_iglob(disp_pattern('left'))

        if self.load_right_label:
            right_disps = natsort_iglob(disp_pattern('right'))
            assert equal_nonempty_length(left_imgs, right_imgs, left_disps, right_disps)
            return [
                dict(
                    imgs=[l, r],
                    gt_disps=[dl, dr]
                )
                for l, r, dl, dr in zip(left_imgs, right_imgs, left_disps, right_disps)
            ]

        assert equal_nonempty_length(left_imgs, right_imgs, left_disps)
        return [
            dict(
                imgs=[l, r],
                gt_disps=[dl]
            )
            for l, r, dl in zip(left_imgs, right_imgs, left_disps)
        ]


@DATASETS.register_module()
class Driving(BaseStereoDataset):
    """Driving dataset for stereo matching.

    A synthetic dataset from the SceneFlow dataset collection containing rendered
    driving scenes with ground truth disparity. The scenes vary in focal length,
    scene type and driving speed.
    """

    METAINFO = {
        'data_set_name': 'driving',
        'task_type': 'standard_stereo_matching',
        'dataset_type': 'synthetic',
        'disp_type': 'dense',
        'has_occlusion': True,
        'disp_max': 192,
        'disp_min': 0,
        'ori_height': 540,
        'ori_width': 960,
        'license': 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License',
        'version': '1.0',
        'url': 'https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html'
    }

    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_root: Optional[str] = '',
                 passes: str = 'frame_cleanpass',
                 focallength: str = '*',
                 scene: str = '*',
                 speed: str = '*',
                 load_right_label: bool = False,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        """Initialize Driving dataset.

        Args:
            ann_file: Path to annotation file.
            data_root: Root path for dataset.
            passes: Data pass type, options: ['frame_cleanpass', 'frame_finalpass'].
            focallength: Focal length setting, use '*' for all.
            scene: Scene type, use '*' for all.
            speed: Driving speed, use '*' for all.
            load_right_label: Whether to load right view labels.
            indices: Indices to subset the dataset.
            serialize_data: Whether to serialize dataset.
            pipeline: Transform pipeline.
            test_mode: Whether in test mode.
            lazy_init: Whether to use lazy initialization.
            max_refetch: Maximum refetch count.
        """
        valid_passes = ['frame_cleanpass', 'frame_finalpass']

        if passes not in valid_passes:
            raise ValueError(f'passes must be one of {valid_passes}')

        self.focallength = focallength
        self.scene = scene
        self.speed = speed

        data_prefix = dict(
            img_left=join_path('Driving', passes),
            img_right=join_path('Driving', passes),
            gt_disp_left=join_path('Driving', 'disparity'),
            gt_disp_right=join_path('Driving', 'disparity'),
        )

        super().__init__(
            ann_file=ann_file,
            metainfo=None,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=None,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            load_right_label=load_right_label,
            lazy_init=lazy_init,
            max_refetch=max_refetch
        )

    def glob_data_list(self) -> List[dict]:
        """Glob and organize stereo image pairs and disparity data.

        Returns:
            List[dict]: List of data samples with image and disparity paths.
        """
        # Driving specific pattern with focallength/scene/speed structure
        img_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'img_{side}'],
            self.focallength,
            self.scene,
            self.speed,
            side,
            '*.png'
        )
        disp_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'gt_disp_{side}'],
            self.focallength,
            self.scene,
            self.speed,
            side,
            '*.pfm'
        )

        left_imgs = natsort_iglob(img_pattern('left'))
        right_imgs = natsort_iglob(img_pattern('right'))
        left_disps = natsort_iglob(disp_pattern('left'))

        if self.load_right_label:
            right_disps = natsort_iglob(disp_pattern('right'))
            assert equal_nonempty_length(left_imgs, right_imgs, left_disps, right_disps)
            return [
                dict(
                    imgs=[l, r],
                    gt_disps=[dl, dr]
                )
                for l, r, dl, dr in zip(left_imgs, right_imgs, left_disps, right_disps)
            ]

        assert equal_nonempty_length(left_imgs, right_imgs, left_disps)
        return [
            dict(
                imgs=[l, r],
                gt_disps=[dl]
            )
            for l, r, dl in zip(left_imgs, right_imgs, left_disps)
        ]


@DATASETS.register_module()
class SubFlyingthings3D(BaseStereoDataset):
    """Subset of FlyingThings3D dataset with filtered samples and occlusion masks."""

    METAINFO = {
        'data_set_name': 'sub_flyingthings3d',
        'task_type': 'standard_stereo_matching',
        'dataset_type': 'synthetic',
        'disp_type': 'dense',
        'has_occlusion': True,
        'disp_max': 192,
        'disp_min': 0,
        'ori_height': 540,
        'ori_width': 960,
        'license': 'Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License',
        'version': '1.0',
        'url': 'https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html'
    }

    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_root: Optional[str] = '',
                 passes: str = 'frame_cleanpass',
                 splits: str = '*',
                 load_right_label: bool = False,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        """Initialize SubFlyingThings3D dataset."""
        valid_passes = ['frame_cleanpass', 'frame_finalpass', '*']
        valid_splits = ['TRAIN', 'TEST', '*']

        if passes not in valid_passes:
            raise ValueError(f'passes must be one of {valid_passes}')
        if splits not in valid_splits:
            raise ValueError(f'splits must be one of {valid_splits}')

        self.passes = passes
        self.splits = splits

        data_prefix = dict(
            img_left=join_path('FlyingThings3D', passes, splits),
            img_right=join_path('FlyingThings3D', passes, splits),
            gt_disp_left=join_path('FlyingThings3D', 'disparity', splits),
            gt_disp_right=join_path('FlyingThings3D', 'disparity', splits),
            occ_mask_left=join_path('FlyingThings3D', 'occlusion', splits),
            occ_mask_right=join_path('FlyingThings3D', 'occlusion', splits)
        )

        super().__init__(
            ann_file=ann_file,
            metainfo=None,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=None,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            load_right_label=load_right_label,
            lazy_init=lazy_init,
            max_refetch=max_refetch
        )

    def glob_data_list(self) -> List[dict]:
        """Glob and organize stereo image pairs, disparity data and occlusion masks."""
        # Check unused files first
        unused_file = join_path(self.data_root, 'FlyingThings3D', 'all_unused_files.txt')
        if not exists(unused_file):
            raise FileNotFoundError(f'all_unused_files.txt not found at {unused_file}')

        # Read and filter unused files
        omits = set(
            line.strip() for line in list_from_file(unused_file)
            if 'left' in line and self.splits in line
        )

        # Create pattern functions
        img_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'img_{side}'],
            f'*/*/{side}/*.png'
        )
        disp_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'gt_disp_{side}'],
            f'*/*/{side}/*.pfm'
        )
        occ_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'occ_mask_{side}'],
            f'{side}/*.png'
        )

        base_path = join_path(self.data_root, 'FlyingThings3D', self.passes) + '/'

        # Get files and sort them naturally
        left_imgs = natsort_iglob(img_pattern('left'))
        right_imgs = natsort_iglob(img_pattern('right'))
        left_disps = natsort_iglob(disp_pattern('left'))

        # Filter out unused files
        valid_indices = []
        for i, img_path in enumerate(left_imgs):
            rel_path = img_path.replace(base_path, '')
            if rel_path not in omits:
                valid_indices.append(i)

        left_imgs = [left_imgs[i] for i in valid_indices]
        right_imgs = [right_imgs[i] for i in valid_indices]
        left_disps = [left_disps[i] for i in valid_indices]

        # Occlusion masks don't need filtering
        left_occs = natsort_iglob(occ_pattern('left'))

        data_list = []
        if self.load_right_label:
            right_disps = [natsort_iglob(disp_pattern('right'))[i] for i in valid_indices]
            right_occs = natsort_iglob(occ_pattern('right'))

            for l_img, r_img, l_disp, r_disp, l_occ, r_occ in zip(
                    left_imgs, right_imgs, left_disps, right_disps, left_occs, right_occs):
                data_list.append(dict(
                    imgs=[l_img, r_img],
                    gt_disps=[l_disp, r_disp],
                    occ_masks=[l_occ, r_occ]
                ))
        else:
            for l_img, r_img, l_disp, l_occ in zip(
                    left_imgs, right_imgs, left_disps, left_occs):
                data_list.append(dict(
                    imgs=[l_img, r_img],
                    gt_disps=[l_disp],
                    occ_masks=[l_occ]
                ))

        return data_list

from typing import List, Union, Optional, Sequence, Callable

from natsort import natsorted

from mmengine.fileio import join_path
from mmdepth.registry import DATASETS
from mmdepth.utils import natsort_iglob
from .base import BaseStereoDataset


@DATASETS.register_module()
class KITTI2015(BaseStereoDataset):
    """KITTI 2015 dataset for stereo matching.

    A real-world dataset containing street scenes with LiDAR ground truth disparity.
    """

    METAINFO = {
        'data_set_name': 'kitti2015',
        'task_type': 'standard_stereo_matching',
        'dataset_type': 'real',
        'disp_type': 'sparse',  # LiDAR ground truth
        'has_occlusion': True,
        'disp_max': 192,
        'disp_min': 0,
        'ori_height': 376,
        'ori_width': 1242,
        'license': 'For academic purpose only',
        'version': '1.0',
        'url': 'http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo'
    }

    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_root: Optional[str] = '',
                 splits: str = 'training',
                 load_right_label: bool = False,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        """Initialize KITTI2015 dataset.

        Args:
            ann_file: Path to annotation file.
            data_root: Root path for dataset.
            splits: Dataset split, options: ['training', 'testing'].
            load_right_label: Whether to load right view labels.
            indices: Indices to subset the dataset.
            serialize_data: Whether to serialize dataset.
            pipeline: Transform pipeline.
            test_mode: Whether in test mode.
            lazy_init: Whether to use lazy initialization.
            max_refetch: Maximum refetch count.
        """
        if splits not in ['training', 'testing']:
            raise ValueError("splits must be one of ['training', 'testing']")

        data_prefix = dict(
            img_left=join_path('KITTI2015', splits, 'image_2'),
            img_right=join_path('KITTI2015', splits, 'image_3'),
            gt_disp_left=join_path('KITTI2015', splits, 'disp_occ_0'),
            gt_disp_right=join_path('KITTI2015', splits, 'disp_occ_1')
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
            '000***_10.png'
        )
        disp_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'gt_disp_{side}'],
            '000***_10.png'
        )

        left_imgs = natsort_iglob(img_pattern('left'))
        right_imgs = natsort_iglob(img_pattern('right'))

        # Only training split has ground truth
        if 'training' in self.data_prefix['img_left']:
            left_disps = natsort_iglob(disp_pattern('left'))

            if self.load_right_label:
                right_disps = natsort_iglob(disp_pattern('right'))
                return [
                    dict(
                        imgs=[l, r],
                        gt_disps=[dl, dr]
                    )
                    for l, r, dl, dr in zip(left_imgs, right_imgs, left_disps, right_disps)
                ]

            return [
                dict(
                    imgs=[l, r],
                    gt_disps=[dl]
                )
                for l, r, dl in zip(left_imgs, right_imgs, left_disps)
            ]

        # Testing split without ground truth
        return [
            dict(
                imgs=[l, r]
            )
            for l, r in zip(left_imgs, right_imgs)
        ]


@DATASETS.register_module()
class KITTI2012(BaseStereoDataset):
    """KITTI 2012 dataset for stereo matching.

    A real-world dataset containing street scenes with LiDAR ground truth disparity.
    """

    METAINFO = {
        'data_set_name': 'kitti2012',
        'task_type': 'standard_stereo_matching',
        'dataset_type': 'real',
        'disp_type': 'sparse',  # LiDAR ground truth
        'has_occlusion': True,
        'disp_max': 192,
        'disp_min': 0,
        'ori_height': 376,
        'ori_width': 1242,
        'license': 'For academic purpose only',
        'version': '1.0',
        'url': 'http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo'
    }

    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_root: Optional[str] = '',
                 splits: str = 'training',
                 colored: bool = True,
                 load_right_label: bool = False,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        """Initialize KITTI2012 dataset.

        Args:
            ann_file: Path to annotation file.
            data_root: Root path for dataset.
            splits: Dataset split, options: ['training', 'testing'].
            colored: Whether to use colored images.
            load_right_label: Whether to load right view labels, must be False
            indices: Indices to subset the dataset.
            serialize_data: Whether to serialize dataset.
            pipeline: Transform pipeline.
            test_mode: Whether in test mode.
            lazy_init: Whether to use lazy initialization.
            max_refetch: Maximum refetch count.
        """
        if splits not in ['training', 'testing']:
            raise ValueError("splits must be one of ['training', 'testing']")

        assert not load_right_label, "KITTI2012 do not contain right view disparity"

        img_type = "colored" if colored else "image"
        data_prefix = dict(
            img_left=join_path('KITTI2012', splits, f'{img_type}_0'),
            img_right=join_path('KITTI2012', splits, f'{img_type}_1'),
            gt_disp_left=join_path('KITTI2012', splits, 'disp_occ')
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
            '000***_10.png'
        )

        left_imgs = natsort_iglob(img_pattern('left'))
        right_imgs = natsort_iglob(img_pattern('right'))

        # Only training split has ground truth
        if 'training' in self.data_prefix['img_left']:
            disp_pattern = join_path(
                self.data_root,
                self.data_prefix['gt_disp_left'],
                '000***_10.png'
            )
            left_disps = natsort_iglob(disp_pattern)

            return [
                dict(
                    imgs=[l, r],
                    gt_disps=[dl]
                )
                for l, r, dl in zip(left_imgs, right_imgs, left_disps)
            ]

        # Testing split without ground truth
        return [
            dict(
                imgs=[l, r]
            )
            for l, r in zip(left_imgs, right_imgs)
        ]
from typing import List, Union, Optional, Sequence, Callable, Literal
import numpy as np
from PIL import Image

from natsort import natsorted

from mmengine.fileio import join_path
from mmdepth.registry import DATASETS
from mmdepth.utils import natsort_iglob
from .base import BaseStereoDataset


@DATASETS.register_module()
class Middlebury2014(BaseStereoDataset):
    """Middlebury 2014 dataset for stereo matching.

    This dataset provides high-resolution stereo pairs with dense ground truth disparities
    and non-occlusion masks. Multiple resolution versions are available.
    """

    METAINFO = {
        'data_set_name': 'middlebury2014',
        'task_type': 'standard_stereo_matching',
        'dataset_type': 'real',
        'disp_type': 'dense',
        'has_occlusion': True,
        'ori_height': None,  # Varies by image and resolution
        'ori_width': None,  # Varies by image and resolution
        'license': 'For academic purpose only',
        'version': '2014',
        'url': 'https://vision.middlebury.edu/stereo/data/scenes2014/'
    }

    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_root: Optional[str] = '',
                 splits: str = 'training',
                 resolution: str = 'H',
                 load_right_label: bool = False,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        """Initialize Middlebury2014 dataset.

        Args:
            ann_file: Path to annotation file.
            data_root: Root path for dataset.
            splits: Dataset split, options: ['training', 'test'].
            resolution: Image resolution version, e.g., 'H' for high resolution.
            load_right_label: Whether to load right view labels.
            indices: Indices to subset the dataset.
            serialize_data: Whether to serialize dataset.
            pipeline: Transform pipeline.
            test_mode: Whether in test mode.
            lazy_init: Whether to use lazy initialization.
            max_refetch: Maximum refetch count.
        """
        if splits not in ['training', 'test']:
            raise ValueError("splits must be one of ['training', 'test']")

        self.splits = splits

        data_prefix = dict(
            img_left=join_path('Midd2014', f'{splits}{resolution}'),
            img_right=join_path('Midd2014', f'{splits}{resolution}')
        )

        if splits == 'training':
            data_prefix.update({
                'gt_disp_left': join_path('Midd2014', f'{splits}{resolution}'),
                'occ_mask_left': join_path('Midd2014', f'{splits}{resolution}')
            })
            if load_right_label:
                data_prefix.update({
                    'gt_disp_right': join_path('Midd2014', f'{splits}{resolution}'),
                    'occ_mask_right': join_path('Midd2014', f'{splits}{resolution}')
                })

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
        """Glob and organize stereo image pairs, disparity maps and occlusion masks.

        Returns:
            List[dict]: List of data samples with image and annotation paths.
        """
        img_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'img_{side}'],
            '*',
            f'im{0 if side == "left" else 1}.png'
        )

        left_imgs = natsort_iglob(img_pattern('left'))
        right_imgs = natsort_iglob(img_pattern('right'))

        # Only training split has ground truth
        if self.splits == 'training':
            disp_pattern = lambda side: join_path(
                self.data_root,
                self.data_prefix[f'gt_disp_{side}'],
                '*',
                f'disp{0 if side == "left" else 1}GT.pfm'
            )
            mask_pattern = lambda side: join_path(
                self.data_root,
                self.data_prefix[f'occ_mask_{side}'],
                '*',
                f'mask{0 if side == "left" else 1}nocc.png'
            )

            left_disps = natsorted(natsort_iglob(disp_pattern('left')))
            left_masks = natsorted(natsort_iglob(mask_pattern('left')))

            if self.load_right_label:
                right_disps = natsorted(natsort_iglob(disp_pattern('right')))
                right_masks = natsorted(natsort_iglob(mask_pattern('right')))
                return [
                    dict(
                        imgs=[l, r],
                        gt_disps=[dl, dr],
                        occ_masks=[lm, rm]
                    )
                    for l, r, dl, dr, lm, rm in zip(
                        left_imgs, right_imgs,
                        left_disps, right_disps,
                        left_masks, right_masks
                    )
                ]

            return [
                dict(
                    imgs=[l, r],
                    gt_disps=[dl],
                    occ_masks=[lm]
                )
                for l, r, dl, lm in zip(
                    left_imgs, right_imgs,
                    left_disps, left_masks
                )
            ]

        # Testing split without ground truth
        return [
            dict(
                imgs=[l, r]
            )
            for l, r in zip(left_imgs, right_imgs)
        ]


@DATASETS.register_module()
class Middlebury2021(BaseStereoDataset):
    """Middlebury 2021 dataset for stereo matching.

    This dataset provides high-resolution stereo pairs with dense ground truth disparities.
    Unlike 2014 version, it doesn't have different resolution versions or occlusion masks.
    """

    METAINFO = {
        'data_set_name': 'middlebury2021',
        'task_type': 'standard_stereo_matching',
        'dataset_type': 'real',
        'disp_type': 'dense',
        'has_occlusion': False,
        'ori_height': None,  # Varies by image
        'ori_width': None,  # Varies by image
        'license': 'For academic purpose only',
        'version': '2021',
        'url': 'https://vision.middlebury.edu/stereo/data/scenes2021/'
    }

    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_root: Optional[str] = '',
                 load_right_label: bool = False,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        """Initialize Middlebury2021 dataset.

        Args:
            ann_file: Path to annotation file.
            data_root: Root path for dataset.
            load_right_label: Whether to load right view labels.
            indices: Indices to subset the dataset.
            serialize_data: Whether to serialize dataset.
            pipeline: Transform pipeline.
            test_mode: Whether in test mode.
            lazy_init: Whether to use lazy initialization.
            max_refetch: Maximum refetch count.
        """
        data_prefix = dict(
            img_left=join_path('Midd2021'),
            img_right=join_path('Midd2021'),
            gt_disp_left=join_path('Midd2021')
        )

        if load_right_label:
            data_prefix['gt_disp_right'] = join_path('Midd2021')

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
        """Glob and organize stereo image pairs and disparity maps.

        Returns:
            List[dict]: List of data samples with image and annotation paths.
        """
        img_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'img_{side}'],
            '*',
            f'im{0 if side == "left" else 1}.png'
        )
        disp_pattern = lambda side: join_path(
            self.data_root,
            self.data_prefix[f'gt_disp_{side}'],
            '*',
            f'disp{0 if side == "left" else 1}.pfm'
        )

        left_imgs = natsort_iglob(img_pattern('left'))
        right_imgs = natsort_iglob(img_pattern('right'))
        left_disps = natsort_iglob(disp_pattern('left'))

        if self.load_right_label:
            right_disps = natsort_iglob(disp_pattern('right'))
            return [
                dict(
                    imgs=[l, r],
                    gt_disps=[dl, dr]
                )
                for l, r, dl, dr in zip(
                    left_imgs, right_imgs,
                    left_disps, right_disps
                )
            ]

        return [
            dict(
                imgs=[l, r],
                gt_disps=[dl]
            )
            for l, r, dl in zip(left_imgs, right_imgs, left_disps)
        ]
from typing import List, Union, Optional, Sequence, Callable
import json

from natsort import natsorted

from mmengine.fileio import join_path
from mmdepth.registry import DATASETS
from mmdepth.utils import natsort_iglob
from .base import BaseStereoDataset


@DATASETS.register_module()
class US3DTrack2(BaseStereoDataset):
    """US3D Track2 dataset for stereo matching.

    Dataset structure:
    - Track2_RGB_{1,2,3,4}/: Contains stereo image pairs and metadata
    - Train_Track2_Truth/: Contains ground truth disparity maps
    """

    METAINFO = {
        'data_set_name': 'us3d_track2',
        'task_type': 'standard_stereo_matching',
        'dataset_type': 'real',  # Assuming this is real-world data
        'disp_type': 'dense',
        'has_occlusion': False,
        'disp_max': None,  # To be determined from data
        'disp_min': 0,
        'ori_height': None,  # Varies by image
        'ori_width': None,  # Varies by image
        'license': 'Refer to US3D dataset terms',
        'version': '1.0'
    }

    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_root: Optional[str] = '',
                 track_parts: Union[str, List[str]] = ['1', '2', '3', '4'],
                 load_right_label: bool = False,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = False,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        """Initialize US3DTrack2 dataset.

        Args:
            ann_file: Path to annotation file (not used).
            data_root: Root path for dataset.
            track_parts: Track parts to include, can be single part or list.
            load_right_label: Whether to load right view labels (not available).
            indices: Indices to subset the dataset.
            serialize_data: Whether to serialize dataset.
            pipeline: Transform pipeline.
            test_mode: Whether in test mode.
            lazy_init: Whether to use lazy initialization.
            max_refetch: Maximum refetch count.
        """
        # Normalize track_parts to list
        if isinstance(track_parts, str):
            track_parts = [track_parts]

        if not all(p in ['1', '2', '3', '4'] for p in track_parts):
            raise ValueError("track_parts must be from ['1', '2', '3', '4']")

        self.track_parts = track_parts

        # Construct data prefix dictionary
        data_prefix = {}
        for part in track_parts:
            data_prefix.update({
                f'img_left_{part}': join_path(f'Track2_RGB_{part}'),
                f'img_right_{part}': join_path(f'Track2_RGB_{part}'),
                f'metadata_{part}': join_path(f'Track2_RGB_{part}')
            })

        # Add ground truth disparity path
        data_prefix['gt_disp_left'] = join_path('Train_Track2_Truth')

        # Right disparity not available
        assert not load_right_label, "US3DTrack2 does not provide right disparity maps"

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
        """Glob and organize stereo image pairs, disparity maps and metadata.

        Returns:
            List[dict]: List of data samples with image paths, disparity paths,
                and metadata paths.
        """
        data_list = []

        for part in self.track_parts:
            # Define patterns for each file type
            left_pattern = join_path(
                self.data_root,
                self.data_prefix[f'img_left_{part}'],
                '*_*_*_LEFT_RGB.tif'
            )
            right_pattern = join_path(
                self.data_root,
                self.data_prefix[f'img_right_{part}'],
                '*_*_*_RIGHT_RGB.tif'
            )
            disp_pattern = join_path(
                self.data_root,
                self.data_prefix['gt_disp_left'],
                '*_*_*_LEFT_DSP.tif'
            )
            meta_pattern = join_path(
                self.data_root,
                self.data_prefix[f'metadata_{part}'],
                '*_*_*_METADATA.json'
            )

            # Get file lists for each type
            left_imgs = natsort_iglob(left_pattern)
            right_imgs = natsort_iglob(right_pattern)
            left_disps = natsort_iglob(disp_pattern)
            meta_files = natsort_iglob(meta_pattern)

            # Organize into samples
            for l, r, d, m in zip(left_imgs, right_imgs, left_disps, meta_files):
                data_list.append(
                    dict(
                        imgs=[l, r],
                        gt_disps=[d],
                        metainfo=m
                    )
                )

        return data_list

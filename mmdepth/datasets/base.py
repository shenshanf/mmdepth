# Copyright (c) MMDepth Contributors. All rights reserved.

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Sequence, Mapping, Callable
from mmengine.config import Config

from mmengine.dataset import BaseDataset


class BaseStereoDataset(BaseDataset, ABC):

    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 load_right_label=False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        self.load_right_label = load_right_label
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch
        )

    def load_data_list(self) -> List[dict]:
        """Load data list from annotation file or by scanning directory.

        If ann_file is specified, load data list from annotation file.
        Otherwise, scan data_root to build data list.

        Returns:
            List[dict]: A list of annotation.
        """
        # If annotation file is provided,
        # the parent class loading method is used
        if self.ann_file is None or self.ann_file != '':
            return super().load_data_list()

        # build data_list by scanning the root folder
        return self.glob_data_list()

    @abstractmethod
    def glob_data_list(self) -> List[dict]:
        """Scan directory to build data list.
        This method should be implemented by subclasses according to their own
        directory structures.

        Returns:
            List[dict]: A list of annotation.
        """
        pass

    # todo: use datasample class in pipline
    # def prepare_data(self, idx) -> Any:
    #     """ override to support datasample
    #     """
    #     data_info = self.get_data_info(idx)
    #     sample_idx = data_info.pop('sample_idx')
    #     data_sample = StereoDataSample(metainfo=dict(file_paths=data_info,
    #                                                  sample_idx=sample_idx))
    #     return self.pipeline(data_sample)


class BaseMonoDatasets(BaseDataset, ABC):
    """Base class for monocular depth estimation datasets.

    This class inherits from BaseDataset and serves as the foundation
    for all monocular depth estimation dataset implementations.
    """

    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Union[Mapping, Config, None] = None,
                 data_root: Optional[str] = '',
                 data_prefix: dict = dict(img_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        """Initialize the BaseMonoDatasets.

        Args:
            ann_file (str, optional): Annotation file path. Default: ''.
            metainfo (dict or Config, optional): Meta information for dataset,
                such as class information. Default: None.
            data_root (str, optional): Data root for images and annotations.
                Default: ''.
            data_prefix (dict): Prefix for image path. Default: dict(img_path='').
            filter_cfg (dict, optional): Config for filtering data. Default: None.
            indices (int or Sequence[int], optional): Support using first few
                data in dataset. Default: None.
            serialize_data (bool): Whether to hold memory using serialized
                objects, when enabled, data loader workers can use shared RAM
                from master process instead of making a copy. Default: True.
            pipeline (list): Processing pipeline. Default: [].
            test_mode (bool): Whether dataset is for testing. Default: False.
            lazy_init (bool): Whether to load annotation during
                instantiation. Default: False.
            max_refetch (int): The maximum number of attempts to retrieve
                a valid sample from the dataset. Default: 1000.
        """
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch
        )

    def load_data_list(self) -> List[dict]:
        """Load data list from annotation file or by scanning directory.

        If ann_file is specified, load data list from annotation file.
        Otherwise, scan data_root to build data list.

        Returns:
            List[dict]: A list of annotation.
        """
        # If annotation file is provided,
        # the parent class loading method is used
        if self.ann_file is None or self.ann_file != '':
            return super().load_data_list()

        # build data_list by scanning the root folder
        return self.glob_data_list()

    @abstractmethod
    def glob_data_list(self) -> List[dict]:
        """Scan directory to build data list.
        This method should be implemented by subclasses according to their own
        directory structures.

        Returns:
            List[dict]: A list of annotation.
        """
        pass

    # todo: use datasample class in pipeline
    # def prepare_data(self, idx) -> Any:
    #     """ override to support datasample
    #     """
    #     data_info = self.get_data_info(idx)
    #     sample_idx = data_info.pop('sample_idx')
    #     data_sample = MonoDepthDataSample(metainfo=dict(file_paths=data_info,
    #                                                     sample_idx=sample_idx))
    #     return self.pipeline(data_sample)

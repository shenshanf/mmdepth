from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Union

from mmengine.evaluator import BaseMetric
from mmdepth.structures import BaseDataSample


class BaseStereoMetric(BaseMetric, ABC):
    default_prefix = 'stereo'

    def __init__(self,
                 gt_key: str = 'gt_disp',
                 pr_key: str = 'pred_disp',
                 eval_region_key: str = 'all',
                 mask_density_thresh: float = 0.0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None):
        """

        Args:
            gt_key:
            pr_key:
            eval_region_key: the key for evaluation region. 'all' for all valid disparity region
            collect_device:
            prefix:
            collect_dir:
        """
        super().__init__(collect_device, prefix, collect_dir)
        self.gt_key = gt_key
        self.pr_key = pr_key
        self.eval_region_key = eval_region_key
        assert 0.0 < mask_density_thresh < 1.0
        self.mask_density_thresh = mask_density_thresh

    @abstractmethod
    def process(self, data_batch: Any, data_samples: Sequence[BaseDataSample]) -> None:
        """
        Args:
            data_batch: not use
            data_samples:

        Returns:

        """
        ...

    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        """

        Args:
            results:

        Returns:

        """
        pass

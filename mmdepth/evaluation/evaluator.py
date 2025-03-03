from typing import Any, Iterator, List, Optional, Sequence, Union

from mmengine.evaluator import Evaluator
from mmdepth.registry import EVALUATOR
from mmdepth.structures import BaseDataSample
# from mmdepth1.datasets.pipline

from .metrics import BaseStereoMetric



@EVALUATOR.register_module()
class StereoEvaluator(Evaluator):
    """
    """

    def process(self,
                data_samples: Sequence[BaseDataSample],
                data_batch: Optional[Any] = None):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            data_samples (Sequence[BaseDataElement]): predictions of the model,
                and the ground truth of the validation set.
            data_batch (Any, optional): A batch of data from the dataloader.
        """
        _data_samples = []
        for data_sample in data_samples:
            # numpy?
            # spatial recover?
            _data_samples.append(data_sample.detach().cpu())

        for metric in self.metrics:
            assert isinstance(metric, BaseStereoMetric)
            metric.process(data_batch, _data_samples)

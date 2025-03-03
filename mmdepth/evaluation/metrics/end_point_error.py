from typing import Any, Sequence, Optional, Union, Tuple
import torch

from mmengine.logging import print_log
from mmdepth.structures import BaseDataSample, DispMap, MultiDispMap
from mmdepth.utils import intersect_masks
from mmdepth.registry import METRICS

from .base import BaseStereoMetric


@METRICS.register_module()
class EPEMetric(BaseStereoMetric):
    """End-Point Error metric for stereo disparity evaluation."""
    # fix: compute in numpy.ndarray field not torch.Tensor field
    default_prefix = 'epe'

    def __init__(self,
                 gt_key: str = 'gt_disp',
                 pr_key: str = 'pred_disp',
                 eval_region_key: str = 'all',
                 mask_density_thresh: float = 0.0,
                 metric_multi: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None):
        super().__init__(gt_key=gt_key, pr_key=pr_key,
                         eval_region_key=eval_region_key,
                         mask_density_thresh=mask_density_thresh,
                         collect_device=collect_device,
                         prefix=prefix, collect_dir=collect_dir)
        self.metric_multi = metric_multi

    @staticmethod
    def epe(gt_map: torch.Tensor, pred_map: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate end-point error between two disparity maps."""
        if mask is not None:
            gt_map = gt_map[mask]
            pred_map = pred_map[mask]
        return torch.abs(gt_map - pred_map)

    @staticmethod
    def _mask_density(mask: Optional[torch.Tensor]) -> float:
        """Calculate density of valid points in mask."""
        if mask is None:  # none mask means dense
            return 1.0
        return (mask.sum().float() / (float(mask.numel()) + 1e-6)).item()

    def eval_predict(self, gt_disp: DispMap, pred_disp: DispMap,
                     eval_region: torch.Tensor, sample_idx: Optional[int]) -> Optional[float]:
        """Evaluate single prediction and return EPE value."""
        eval_mask = intersect_masks(gt_disp.v_mask, pred_disp.v_mask, eval_region)
        mask_density = self._mask_density(eval_mask)

        if mask_density >= self.mask_density_thresh:
            return self.epe(gt_disp.map, pred_disp.map, eval_mask).mean().item()

        msg = f"mask_density:{mask_density:.1%}"
        msg = f"skip at idx: {sample_idx}, {msg}" if sample_idx is not None else f"skip, {msg}"
        print_log(msg, level=30)
        return None

    def _unpack(self, data_sample: BaseDataSample) -> Tuple[DispMap, Union[DispMap, MultiDispMap],
    torch.Tensor, Optional[int]]:
        """Unpack necessary data from data sample."""
        gt_disp = data_sample.get(self.gt_key)
        if gt_disp is None:
            raise ValueError(f"Ground truth key '{self.gt_key}' not found")
        if not isinstance(gt_disp, DispMap):
            raise TypeError(f"Ground truth must be DispMap, got {type(gt_disp)}")

        pr_disp = data_sample.get(self.pr_key)
        if pr_disp is None:
            raise ValueError(f"Prediction key '{self.pr_key}' not found")
        if not isinstance(pr_disp, (DispMap, MultiDispMap)):
            raise TypeError(f"Prediction must be DispMap or MultiDispMap, got {type(pr_disp)}")

        eval_region = None if self.eval_region_key == 'all' else data_sample.get(self.eval_region_key)
        if self.eval_region_key != 'all' and eval_region is None:
            raise ValueError(f"Evaluation region '{self.eval_region_key}' not found")

        sample_idx = data_sample.get("sample_idx", None)
        return gt_disp, pr_disp, eval_region, sample_idx

    def _process(self, data_sample: BaseDataSample) -> None:
        """Process single data sample."""
        gt_disp, pr_disp, eval_region, sample_idx = self._unpack(data_sample)

        if isinstance(pr_disp, DispMap) or (isinstance(pr_disp, MultiDispMap) and not self.metric_multi):
            pred_disp = pr_disp if isinstance(pr_disp, DispMap) else pr_disp[-1]
            metric = self.eval_predict(gt_disp, pred_disp, eval_region, sample_idx)
            if metric is not None:
                self.results.append(metric)

        elif isinstance(pr_disp, MultiDispMap) and self.metric_multi:
            metric_list = []
            for pred_disp in pr_disp.values():
                metric = self.eval_predict(gt_disp, pred_disp, eval_region, sample_idx)
                if metric is not None:
                    metric_list.append(metric)
            if metric_list:
                self.results.append(metric_list)

        else:
            raise TypeError(f"Unsupported prediction type: {type(pr_disp)}")

    def process(self, data_batch: Any, data_samples: Sequence[BaseDataSample]) -> None:
        """Process batch of samples."""
        for data_sample in data_samples:
            self._process(data_sample)

    def compute_metrics(self, results: list) -> dict:
        """Compute final metrics from results.
        note: will log at terminal and visual_backend at
             `mmdepth1.engine.LoggerHook.after_val_epoch`
        """

        # compute total metric in whole eval loop
        total_metric = torch.asarray(results).mean(dim=0).tolist()

        if isinstance(total_metric, list):
            return {f"{self.pr_key}/{self.prefix}_{self.eval_region_key}_s{idx}": _metric
                    for idx, _metric in enumerate(total_metric)}

        return {f"{self.pr_key}/{self.prefix}_{self.eval_region_key}": total_metric}

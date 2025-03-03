from typing import Optional
import torch

from mmengine.logging import print_log
from mmdepth.structures import BaseDataSample, DispMap, MultiDispMap
from mmdepth.utils import intersect_masks
from mmdepth.registry import METRICS
from .end_point_error import EPEMetric


@METRICS.register_module()
class KittiD1Metric(EPEMetric):
    """KITTI D1 Error metric for stereo disparity evaluation.

    Calculate the percentage of pixels where the disparity error exceeds either:
    - the absolute threshold (default 3 pixels) OR
    - the relative threshold (default 5% of ground truth)
    """

    default_prefix = 'd1'

    def __init__(self,
                 gt_key: str = 'gt_disp',
                 pr_key: str = 'pred_disp',
                 eval_region_key: str = 'all',
                 mask_density_thresh: float = 0.0,
                 metric_multi: bool = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None,
                 abs_thresh: float = 3.0,  # 3 pixels threshold
                 rel_thresh: float = 0.05):  # 5% threshold
        super().__init__(gt_key=gt_key,
                         pr_key=pr_key,
                         eval_region_key=eval_region_key,
                         mask_density_thresh=mask_density_thresh,
                         metric_multi=metric_multi,
                         collect_device=collect_device,
                         prefix=prefix,
                         collect_dir=collect_dir)
        assert abs_thresh > 0
        assert 0 < rel_thresh < 1
        self.abs_thresh = abs_thresh
        self.rel_thresh = rel_thresh

    def d1(self, gt_disp: torch.Tensor, pr_disp: torch.Tensor, mask: Optional[torch.Tensor]):
        if mask is not None:
            gt_disp = gt_disp[mask]
            pr_disp = pr_disp[mask]
        epe = torch.abs(gt_disp - pr_disp)

        d1_mask = (epe > self.abs_thresh) & (epe / torch.abs(pr_disp) > self.rel_thresh)
        return d1_mask.float()

    def eval_predict(self, gt_disp: DispMap, pred_disp: DispMap,
                     eval_region: torch.Tensor, sample_idx: Optional[int]) -> Optional[float]:
        """Evaluate single prediction and return EPE value."""
        eval_mask = intersect_masks(gt_disp.v_mask, pred_disp.v_mask, eval_region)
        mask_density = self._mask_density(eval_mask)

        if mask_density >= self.mask_density_thresh:
            return self.d1(gt_disp.map, pred_disp.map, eval_mask).mean().item()

        msg = f"mask_density:{mask_density:.1%}"
        msg = f"skip at idx: {sample_idx}, {msg}" if sample_idx is not None else f"skip, {msg}"
        print_log(msg, level=30)
        return None

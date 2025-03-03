from typing import Optional
import torch

from mmengine.logging import print_log
from mmdepth.structures import BaseDataSample, DispMap
from mmdepth.utils import intersect_masks
from mmengine.registry import METRICS

from .end_point_error import EPEMetric


@METRICS.register_module()
class EPXMetric(EPEMetric):
    """End-Point Error Threshold metric for stereo disparity evaluation.

    This metric calculates the percentage of pixels where the disparity error
    exceeds a specified threshold (default 3 pixels). It can be seen as a
    generalization of the error threshold metric, where EPE > threshold is
    considered as an error.
    """

    default_prefix = 'epx'

    def __init__(self,
                 gt_key: str = 'gt_disp',
                 pr_key: str = 'pred_disp',
                 eval_region_key: str = 'all',
                 mask_density_thresh: float = 0.0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None,
                 error_thresh: float = 3.0):
        """Initialize EPXMetric.
        """
        super().__init__(gt_key=gt_key,
                         pr_key=pr_key,
                         eval_region_key=eval_region_key,
                         mask_density_thresh=mask_density_thresh,
                         collect_device=collect_device,
                         prefix=prefix,
                         collect_dir=collect_dir)
        assert error_thresh > 0, "Error threshold must be positive"
        self.error_thresh = error_thresh

    def epx(self,
            gt_disp: torch.Tensor,
            pr_disp: torch.Tensor,
            mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Calculate percentage of pixels with error above threshold.
        """
        epe = self.epe(gt_disp, pr_disp, mask)
        epx_mask = epe > self.error_thresh
        return epx_mask.float()

    def eval_predict(self, gt_disp: DispMap, pred_disp: DispMap,
                     eval_region: torch.Tensor, sample_idx: Optional[int]) -> Optional[float]:
        """"""
        eval_mask = intersect_masks(gt_disp.v_mask, pred_disp.v_mask, eval_region)
        mask_density = self._mask_density(eval_mask)

        if mask_density >= self.mask_density_thresh:
            return self.epx(gt_disp.map, pred_disp.map, eval_mask).mean().item()

        msg = f"mask_density:{mask_density:.1%}"
        msg = f"skip at idx: {sample_idx}, {msg}" if sample_idx is not None else f"skip, {msg}"
        print_log(msg, level=30)
        return None

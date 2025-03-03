from typing import Any, Sequence, Optional, Tuple, Union
import torch
import numpy as np
from pathlib import Path

from mmengine.logging import print_log
from mmengine.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.utils import mkdir_or_exist

from mmdepth.structures import BaseDataSample, DispMap, MultiDispMap
from mmdepth.utils import intersect_masks
from mmdepth.fileio import write_pfm


@METRICS.register_module()
class DumpStereoResults(BaseMetric):
    """Dump EPE maps to PFM format files."""

    def __init__(self,
                 out_dir: str,
                 gt_key: str = 'gt_disp',
                 pr_key: str = 'pred_disp',
                 invalid_value: float = np.nan,
                 dump_multi: bool = True,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        mkdir_or_exist(out_dir)
        self.out_dir = Path(out_dir)
        self.gt_key = gt_key
        self.pr_key = pr_key
        self.invalid_value = invalid_value
        self.dump_multi = dump_multi

    def epe_map(self,
                gt_disp: torch.Tensor,
                pr_disp: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if gt_disp.shape != pr_disp.shape:
            raise ValueError(f"Shape mismatch: gt_disp {gt_disp.shape} vs pr_disp {pr_disp.shape}")

        epe_map = torch.abs(gt_disp - pr_disp)
        if mask is not None:
            if mask.shape != epe_map.shape:
                raise ValueError(f"Mask shape {mask.shape} doesn't match disparity shape {epe_map.shape}")
            epe_map[~mask] = self.invalid_value

        return epe_map

    def dump_predict(self,
                     gt_disp: DispMap,
                     pred_disp: DispMap,
                     sample_idx: Optional[int],
                     batch_idx: int = 0,
                     level_idx: Optional[int] = None) -> None:
        eval_mask = intersect_masks(gt_disp.v_mask, pred_disp.v_mask)
        epe_map = self.epe_map(gt_disp.map, pred_disp.map, eval_mask)

        sample_str = f"_{sample_idx}" if sample_idx is not None else ""
        level_str = f"_s{level_idx}" if level_idx is not None else ""

        epe_filename = self.out_dir / f"epe{sample_str}/epe_b{batch_idx}{level_str}.pfm"
        disp_filename = self.out_dir / f"disp{sample_str}/disp_b{batch_idx}{level_str}.pfm"

        try:
            write_pfm(epe_filename, epe_map.cpu().numpy())
        except Exception as e:
            print_log(f"Failed to save EPE map to {epe_filename}: {str(e)}",
                      logger='current', level=40)

        try:
            gt_disp_map = gt_disp.cpu().numpy().map_clone(self.invalid_value)
            write_pfm(disp_filename, gt_disp_map)
        except Exception as e:
            print_log(f"Failed to save EPE map to {epe_filename}: {str(e)}",
                      logger='current', level=40)

    def _unpack(self, data_sample: BaseDataSample) -> Tuple[DispMap, Union[DispMap, MultiDispMap], Optional[int]]:
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

        return gt_disp, pr_disp, data_sample.get("sample_idx", None)

    def _process(self, data_sample: BaseDataSample, batch_idx: int = 0) -> None:
        gt_disp, pr_disp, sample_idx = self._unpack(data_sample)

        if isinstance(pr_disp, DispMap) or (isinstance(pr_disp, MultiDispMap) and not self.dump_multi):
            pred_disp = pr_disp if isinstance(pr_disp, DispMap) else pr_disp[-1]
            self.dump_predict(gt_disp, pred_disp, sample_idx, batch_idx)
        elif isinstance(pr_disp, MultiDispMap) and self.dump_multi:
            for level_idx, pred_disp in enumerate(pr_disp.values()):
                self.dump_predict(gt_disp, pred_disp, sample_idx, batch_idx, level_idx)
        else:
            raise TypeError(f"Unsupported prediction type: {type(pr_disp)}")

    def process(self, data_batch: Any, data_samples: Sequence[BaseDataSample]) -> None:
        for batch_idx, data_sample in enumerate(data_samples):
            self._process(data_sample, batch_idx)

    def compute_metrics(self, results: list) -> dict:
        print_log(f'EPE maps have been saved to {self.out_dir}', logger='current')
        return {}

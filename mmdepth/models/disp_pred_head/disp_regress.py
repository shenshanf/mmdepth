from typing import Union, Dict, List
import torch
from mmdepth.registry import PREDICT_HEADERS, LOSSES
from mmdepth.models.structures import CostVolume, DispMap
from mmengine.model import BaseModule
from .base_disp_head import BaseDispHead
from mmdepth.utils import multi_apply


# from mmdepth1.models.modules.losses import


@PREDICT_HEADERS.register_module()
class DispRegressHead(BaseDispHead):
    gt_disp_left_key = 'gt_disp_left'
    loss_pattern = "regress_loss{_s[%d]}"
    pred_pattern = "pred_disp{_s[%d]}"

    def __init__(self, loss: Dict, softmax: bool = True, keep_dim: bool = True):
        """

        Args:
            loss:
            softmax: False means already softmax outside
            keep_dim:
        """
        super().__init__()
        self.loss_fn = LOSSES.build(loss)
        self.softmax = softmax
        self.keep_dim = keep_dim

    @multi_apply
    def forward(self, prob_volume: CostVolume) -> DispMap:
        """Forward pass of disparity classifier.

        note: Important distinction about sample_grid:
            - Relative coordinates: Represents offsets from disparity prior. The classifier outputs
              disparity updates that need to be residually connected with the prior.
            - Absolute coordinates: Represents positions in search space. The classifier directly
              outputs disparity values.

        Args:
            prob_volume: Cost volume containing:
                data: Probability volume of shape [b,d,h,w] or [b,1,d,h,w]
                sample_grid: Sample coordinates of shape [b,d,h,w] or [b=1,d,h=1,w=1]

        Returns:
            Predicted disparity map
        """
        prob_data = prob_volume.data
        sample_grid = prob_volume.sample_grid

        # Squeeze singleton channel dimension if 5D input
        # [b,d,h,w] <- [b,1,d,h,w]
        if prob_data.ndim == 5:
            assert prob_data.shape[1] == 1
            prob_data = prob_data.squeeze(dim=1)

        if self.softmax:
            # if not, the prob_data is already 'softmax' outside
            prob_data = prob_data.softmax(dim=1)

        # Compute expected disparity through soft argmax
        # pr_disp = torch.sum(prob_data * sample_grid, dim=1, keepdim=self.keep_dim)
        pr_disp = torch.einsum('bdhw, bdhw->bhw', prob_data, sample_grid)
        if self.keep_dim:
            pr_disp = pr_disp.unsqueeze(1)

        return DispMap(data=pr_disp, v_mask=None)

    def loss_by_results(self, results: Union[DispMap, List[DispMap]], datasamples) -> Dict[str, torch.Tensor]:
        """ todo: 重构一下datasample

        Args:
            results:
            datasamples:

        Returns:

        """
        gt_disp = datasamples.get(self.gt_disp_left_key)
        if gt_disp is None:
            raise ValueError(f"Ground truth disparity '{self.gt_disp_left_key}' not found")

        if isinstance(results, list):
            return {self.loss_pattern % i: self.loss_fn(res.data, gt_disp)
                    for i, res in enumerate(results)}
        return {self.loss_pattern.split('{')[0]: self.loss_fn(results.data, gt_disp)}

    def pack_results(self, results: Union[DispMap, List[DispMap]], datasamples):
        """

        Args:
            results:
            datasamples:

        Returns:

        """
        if isinstance(results, list):
            for i, res in enumerate(results):
                datasamples[self.pred_pattern % i] = res.data
        else:
            datasamples[self.pred_pattern.split('{')[0]] = results.data
        return datasamples

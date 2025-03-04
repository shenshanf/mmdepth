from mmengine.model import BaseModule
from mmdepth.registry import COST_VOLUMES


@COST_VOLUMES.register_module()
class CostVolBuilderWrapper(BaseModule):
    """wrapper for building multi-scale cost volumes"""
    def __init__(self, cost_builder_cfgs):
        super().__init__()
        builders = []
        for cost_builder_cfg in cost_builder_cfgs:
            builders.append(COST_VOLUMES.buid(cost_builder_cfg))
        self.builders = builders

    def __len__(self):
        return len(self.builders)

    def forward(self, feat_pairs):
        assert len(feat_pairs) == len(self.builders)

        cost_vols = []
        for feat_pair, builder in zip(feat_pairs, self.builders):
            feat_left, feat_right = feat_pairs
            cost_vols.append(builder(feat_left, feat_right))
        return cost_vols

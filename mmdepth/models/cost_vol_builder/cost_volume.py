from mmdepth.registry import COST_VOLUMES
from mmdepth.utils import BackendManager

from .pytorch import CorrCostVolBuilder as TH_CorrCostVolBuilder
from .pytorch import GwcCostVolBuilder as TH_GwcCostVolBuilder
from .pytorch import ConcatCostVolBuilder as TH_ConcatCostVolBuilder
from .pytorch import CostVolBuilder as TH_CostVolBuilder
from .pytorch import DymCostVolBuilder as TH_DymCostVolBuilder
from .pytorch import UniformCostVolBuilder as TH_UniformCostVolBuilder
from .pytorch import DymUniformCostVolBuilder as TH_DymUniformCostVolBuilder
from .pytorch import CostVolLookUp as TH_CostVolLookUp
from .pytorch import CostVolSliceYield as TH_CostVolSliceYield


@COST_VOLUMES.register_module()
@BackendManager()
class CorrCostVolBuilder(TH_CorrCostVolBuilder):
    """Implemented in `(backend).CorrCostVolBuilder`, e.g. `.pytorch.CorrCostVolBuilder`
    """

    def __init__(self, *args, backend='pytorch', **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend


@COST_VOLUMES.register_module()
@BackendManager()
class GwcCostVolBuilder(TH_GwcCostVolBuilder):
    """Implemented in `(backend).GwcCostVolBuilder`, e.g. `.pytorch.GwcCostVolBuilder`
    """

    def __init__(self, *args, backend='pytorch', **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend


@COST_VOLUMES.register_module()
@BackendManager()
class ConcatCostVolBuilder(TH_ConcatCostVolBuilder):
    """Implemented in `(backend).ConcatCostVolBuilder`, e.g. `.pytorch.ConcatCostVolBuilder`
    """

    def __init__(self, *args, backend='pytorch', **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend


@COST_VOLUMES.register_module()
@BackendManager()
class CostVolBuilder(TH_CostVolBuilder):
    """
    """

    def __init__(self, *args, backend='pytorch', **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend


@COST_VOLUMES.register_module()
@BackendManager()
class DymCostVolBuilder(TH_DymCostVolBuilder):
    def __init__(self, *args, backend='pytorch', **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend


@COST_VOLUMES.register_module()
@BackendManager()
class UniformCostVolBuilder(TH_UniformCostVolBuilder):
    def __init__(self, *args, backend='pytorch', **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend


@COST_VOLUMES.register_module()
@BackendManager()
class DymUniformCostVolBuilder(TH_DymUniformCostVolBuilder):
    def __init__(self, *args, backend='pytorch', **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend


@COST_VOLUMES.register_module()
@BackendManager()
class CostVolLookUp(TH_CostVolLookUp):
    def __init__(self, *args, backend='pytorch', **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend


@COST_VOLUMES.register_module()
@BackendManager()
class CostVolSliceYield(TH_CostVolSliceYield):
    def __init__(self, *args, backend='pytorch', **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend

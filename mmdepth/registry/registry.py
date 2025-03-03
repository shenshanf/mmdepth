from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry

from deprecated import deprecated

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', parent=MMENGINE_RUNNERS)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', parent=MMENGINE_RUNNER_CONSTRUCTORS)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', parent=MMENGINE_LOOPS)
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook',
    parent=MMENGINE_HOOKS,
    locations=['mmdepth.engine.hooks'])

# manage data-related modules
DATASETS = Registry(
    'dataset',
    parent=MMENGINE_DATASETS,
    locations=['mmdepth.datasets'])

TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['mmdepth.datasets.pipline', 'mmdepth.visualization'])

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmdepth.models'])

BACKBONES = MODELS
NECKS = MODELS
HEADERS = MODELS

# sub models
FEATURE_ENCODERS = MODELS
COST_VOLUMES = Registry('cost_volume', scope='mmdepth',
                        locations=['mmdepth.models.modules.cost_vol_builder'])
LOSSES = Registry('loss', scope='mmdepth',
                  locations=['mmdepth.models.modules.losses'])

AGGREGATORS = MODELS
PREDICT_HEADERS = MODELS
SUB_MODELS = MODELS

# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['mmdepth.models'])
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['mmdepth.models'])

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    locations=['mmdepth.engine.optimizer'])

# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    locations=['mmdepth.engine.optim_wrapper'])

# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['mmdepth.engine.optim_wrapper_constructor'])
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=['mmdepth.engine.scheduler'])
# manage all kinds of metrics
METRICS = Registry(
    'metric',
    parent=MMENGINE_METRICS,
    locations=['mmdepth.evaluation'])
# manage evaluator
EVALUATOR = Registry(
    'evaluator',
    parent=MMENGINE_EVALUATOR,
    locations=['mmdepth.evaluation'])

#
# modules below, therefore locations are not specified for Registry.

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry('task util', parent=MMENGINE_TASK_UTILS)

# manage visualizer
VISUALIZERS = Registry('visualizer', parent=MMENGINE_VISUALIZERS)
# manage visualizer backend
VISBACKENDS = Registry('vis_backend', parent=MMENGINE_VISBACKENDS)

# manage logprocessor
LOG_PROCESSORS = Registry('log_processor', parent=MMENGINE_LOG_PROCESSORS)

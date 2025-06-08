from ..utils import register_module
from .metric import (
    MetricWrap,
    CrossEntropyLoss,
    CrossEntropyLossGrouped,
    MSELoss,
    LPIPSLoss,
    ARI,
    mBO,
    mIoU,
)
from .optim import Adam, GradScaler, ClipGradNorm, ClipGradValue
from .callback import Callback
from .callback_log import AverageLog, SaveModel
from .callback_sched import (
    CbLinear,
    CbCosine,
    CbLnCosine,
    CbCosineLinear,
    CbLinearCosine,
    CbSquarewave,
)

[register_module(_) for _ in locals().values() if isinstance(_, type)]

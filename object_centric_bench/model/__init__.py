from ..utils import register_module
from .basic import (
    ModelWrap,
    Sequential,
    ModuleList,
    Embedding,
    Conv2d,
    PixelShuffle,
    ConvTranspose2d,
    Interpolate,
    Linear,
    Dropout,
    AdaptiveAvgPool2d,
    GroupNorm,
    LayerNorm,
    ReLU,
    GELU,
    SiLU,
    Mish,
    MultiheadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    CNN,
    MLP,
    Identity,
    DINO2ViT,
    EncoderTAESD,
    DecoderTAESD,
)
from .ocl import (
    SlotAttention,
    NormalShared,
    NormalSeparat,
    CartesianPositionalEmbedding2d,
    LearntPositionalEmbedding,
    VQVAE,
    Codebook,
    LearntPositionalEmbedding,
)
from .slatesteve import SLATE, STEVE, ARTransformerDecoder
from .slotdiffusion import (
    SlotDiffusion,
    ConditionDiffusionDecoder,
    NoiseSchedule,
    UNet2dCondition,
)
from .vaez import VQVAEZ, QuantiZ, VQVAEZGrouped, VQVAEZMultiScale

[register_module(_) for _ in locals().values() if isinstance(_, type)]

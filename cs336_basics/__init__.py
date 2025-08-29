import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

# 从bpe_tokenizer子包导入所有模块
from .bpe_tokenizer import *
from .transformer.module import Linear
from .transformer.module import Embedding
from .transformer.module import RMSNorm
from .transformer.module import PositionWiseFFN
from .transformer.module import RoPE
from .transformer.module import softmax
from .transformer.module import scaled_dot_product_attention
from .transformer.module import MultiheadSelfAttention
from .transformer.module import MultiheadSelfAttentionWithRoPE
from .transformer.module import TransformerBlock
from .transformer.module import TransformerLM

__all__ = [
    "BPETokenizer",
    "train_bpe",
    "parallel_preprocess_from_file",
    "get_pre_token_freq",
    "init_byte_cache",
    "get_byte_value",
    "create_byte_pair",
    "merge_bytes",
    "get_pair_cached",
    "Linear",
    "Embedding",
    "RMSNorm",
    "PositionWiseFFN",
    "RoPE",
    "softmax",
    "scaled_dot_product_attention",
    "MultiheadSelfAttention",
    "MultiheadSelfAttentionWithRoPE",
    "TransformerBlock",
    "TransformerLM",
]

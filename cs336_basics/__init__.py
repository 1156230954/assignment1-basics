import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

# 从bpe_tokenizer子包导入所有模块
from .bpe_tokenizer import *

__all__ = [
    "BPETokenizer",
    "train_bpe",
    "parallel_preprocess_from_file",
    "get_pre_token_freq",
    "init_byte_cache",
    "get_byte_value",
    "create_byte_pair",
    "merge_bytes",
    "get_pair_cached"
]

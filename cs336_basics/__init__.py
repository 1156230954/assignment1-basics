import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

# 导入BPE分词器相关模块
from .train_bpe import train_bpe, parallel_preprocess_from_file
from .bytes_utils import (
    init_byte_cache,
    get_byte_value,
    create_byte_pair,
    merge_bytes,
    get_pair_cached,
)

__all__ = [
    "BPETokenizer",
    "get_tokenizer", 
    "train_bpe",
    "parallel_preprocess_from_file",
    "init_byte_cache",
    "get_byte_value",
    "create_byte_pair",
    "merge_bytes",
    "get_pair_cached"
]

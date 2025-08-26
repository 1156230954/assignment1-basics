"""
字节处理工具模块
包含BPE训练和分词中使用的字节操作函数
"""

# 预计算字节类型映射，避免重复的isinstance检查
_byte_cache = {}

def init_byte_cache():
    """初始化字节缓存，将int类型的字节值预转换为bytes对象"""
    global _byte_cache
    _byte_cache.clear()
    for i in range(256):
        _byte_cache[i] = bytes([i])

def get_byte_value(byte_val):
    """获取字节值，如果输入是int则转换为bytes，否则直接返回"""
    return _byte_cache.get(byte_val, byte_val)

def create_byte_pair(token_bytes, i):
    """创建字节对，使用缓存优化"""
    a = _byte_cache.get(token_bytes[i], token_bytes[i])
    b = _byte_cache.get(token_bytes[i+1], token_bytes[i+1])
    return (a, b)

def merge_bytes(token_bytes, i):
    """合并两个字节，使用缓存优化"""
    a = _byte_cache.get(token_bytes[i], token_bytes[i])
    b = _byte_cache.get(token_bytes[i+1], token_bytes[i+1])
    return a + b

# 缓存常用的字节对，避免重复创建
_pair_cache = {}

def get_pair_cached(token_bytes, i):
    """带缓存的字节对获取函数"""
    key = (token_bytes[i], token_bytes[i+1])
    if key not in _pair_cache:
        _pair_cache[key] = (bytes([token_bytes[i]]), bytes([token_bytes[i+1]]))
    return _pair_cache[key]

import re
import multiprocessing as mp
from collections import defaultdict
import time

from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.bytes_utils import init_byte_cache, create_byte_pair, merge_bytes


# 正则表达式，用于预分词
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z\u00C0-\u017F]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""

# 初始化词汇表
# 因为训练的是Byte级别的BPE，所以初始词汇表是256个字节，即一个Byte的大小
def init_vocab(special_tokens):
    # 256个字节作为初始词汇
    vocab = {i: bytes([i]) for i in range(256)}
    # 添加特殊令牌（ID从256开始）
    for idx, token in enumerate(special_tokens, start=256):
        vocab[idx] = token.encode("utf-8")  # 特殊令牌转为字节
    return vocab




# 统计所有相邻字节对的频率（考虑预令牌出现次数）
def get_pair_frequencies(pre_token_freq) :
    pair_freq = defaultdict(int)
    for token_bytes, count in pre_token_freq.items():
        # 遍历令牌内的相邻字节对（如(l,o,w)→(l,o)、(o,w)）
        for i in range(len(token_bytes) - 1):
            pair = create_byte_pair(token_bytes, i)
            pair_freq[pair] += count  # 累加频率（预令牌出现次数×1）
    return pair_freq

def train_bpe(
    vocab_size,
    special_tokens,
    pre_token_freq
):
    """
    训练BPE分词器：
    返回：词汇表（ID→字节）和合并历史
    """
    # 初始化字节缓存，避免重复的isinstance检查
    init_byte_cache()
    
    # 初始化词汇表
    vocab = init_vocab(special_tokens)
    current_vocab_size = len(vocab)
    if current_vocab_size >= vocab_size:
        return vocab, []  # 无需合并
    
    merges = []  # 记录合并历史（按顺序）

    # 初始化字节对频率（只计算一次全量）
    pair_freq = get_pair_frequencies(pre_token_freq)
    
    # 迭代合并，直到达到目标词汇表大小
    while current_vocab_size < vocab_size:
        if not pair_freq:
            break 
        
        # 1. 选择频率最高的对（频率相同则选词法更大的）
        best_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
        
        # 2. 执行合并并记录受影响的令牌
        new_pre_token_freq = defaultdict(int)
        affected_old_tokens = []  # 记录被合并操作影响的旧令牌
        
        for token_bytes, count in pre_token_freq.items():
            # 检查当前令牌是否包含最佳合并对
            contains_best_pair = False
            for i in range(len(token_bytes) - 1):
                if create_byte_pair(token_bytes, i) == best_pair:
                    contains_best_pair = True
                    break
            
            if not contains_best_pair:
                new_pre_token_freq[token_bytes] += count
                continue
            
            # 记录受影响的旧令牌
            affected_old_tokens.append((token_bytes, count))
            
            # 执行合并操作
            new_tokens = []
            i = 0
            while i < len(token_bytes):
                if i < len(token_bytes) - 1 and create_byte_pair(token_bytes, i) == best_pair:
                    merged = merge_bytes(token_bytes, i)
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(token_bytes[i])
                    i += 1
            new_token_bytes = tuple(new_tokens)
            new_pre_token_freq[new_token_bytes] += count
        
        # 3. 增量更新字节对频率
        # 3.1 移除受影响旧令牌中的字节对
        for token_bytes, count in affected_old_tokens:
            for i in range(len(token_bytes) - 1):
                pair = create_byte_pair(token_bytes, i)
                pair_freq[pair] -= count
                if pair_freq[pair] <= 0:
                    del pair_freq[pair]
        
        # 3.2 添加新令牌中产生的字节对
        for token_bytes, count in new_pre_token_freq.items():
            # 只处理新生成的令牌（旧令牌已保留，无需重复计算）
            if token_bytes not in pre_token_freq:
                for i in range(len(token_bytes) - 1):
                    pair = create_byte_pair(token_bytes, i)                    
                    pair_freq[pair] += count

        # 4. 更新词汇表和合并历史
        pre_token_freq = new_pre_token_freq
        new_token = best_pair[0] + best_pair[1]
        vocab[current_vocab_size] = new_token
        merges.append(best_pair)
        current_vocab_size += 1
        
    
    return vocab,merges



# 并行预分词主函数
def parallel_preprocess_from_file(
    file_path,
    special_tokens,
    desired_num_chunks
):
    # 特殊令牌的字节形式（用于分块）
    split_special_token = special_tokens[0].encode("utf-8")

    # 1. 获取分块边界
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, split_special_token)
    # 生成(start, end)对
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))

    # 2. 多进程处理分块
    # 传递文件路径而非文件对象（进程间无法共享文件对象）
    with mp.Pool(processes=len(chunk_ranges)) as pool:
        # 为每个分块创建任务
        tasks = [
            (file_path, start, end, special_tokens)
            for start, end in chunk_ranges
        ]
        # 并行执行
        results = pool.starmap(pre_tokenize_file, tasks)

    # 3. 合并所有分块的频率
    total_freq = defaultdict(int)
    for freq in results:
        for token_bytes, count in freq.items():
            total_freq[token_bytes] += count
    return total_freq

# 单进程处理分块的预分词逻辑
def pre_tokenize_file(
    file_path,
    start,
    end,
    special_tokens
):
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
    return get_pre_token_freq(chunk_text, special_tokens)


def get_pre_token_freq(
    text,
    special_tokens
):
    # 获取预分词列表
    pre_tokens = pre_tokenize(text, special_tokens)
    
    # 统计频率
    pre_token_freq = defaultdict(int)
    for token_bytes in pre_tokens:
        pre_token_freq[token_bytes] += 1
    
    return pre_token_freq

def pre_tokenize_iter(texts, special_tokens):
    sorted_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
    special_pattern = special_pattern = "|".join(re.escape(token) for token in sorted_tokens) if special_tokens else r"(?!)"
    for text in texts:
            # 首先按特殊token进行分割
            text_blocks = re.split(f'({special_pattern})', text)
            
            for block in text_blocks:
                if block in special_tokens:
                    # 特殊token直接生成
                    yield block.encode('utf-8')
                elif block:  # 跳过空字符串
                    # 对普通文本使用word_pattern进行分词并生成
                    for match in re.finditer(PAT, block):
                        yield match.group(0).encode('utf-8')

def pre_tokenize(text, special_tokens):
    # 按特殊令牌拆分，避免跨令牌合并
    sorted_tokens = sorted(special_tokens, key=lambda x: len(x), reverse=True)
    special_pattern = special_pattern = "|".join(re.escape(token) for token in sorted_tokens) if special_tokens else r"(?!)"
    text_blocks = re.split(f'({special_pattern})', text)

    pre_tokens = []
    for block in text_blocks:
        if block in special_tokens:
            pre_tokens.append(block.encode('utf-8'))
            continue
        for match in re.finditer(PAT, block):
            pre_token = match.group()
            # 转换为字节序列（用于后续BPE合并）
            token_bytes = pre_token.encode("utf-8")
            pre_tokens.append(token_bytes)
    
    return pre_tokens
    

if __name__ == "__main__":
    import cProfile
    import pstats
    
    # 示例：使用并行预分词训练BPE
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"  # 数据集文件路径
    special_tokens = ["<|endoftext|>"]
    
    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 并行预分词
    start_time = time.time()
    pre_token_freq = parallel_preprocess_from_file(
        file_path=file_path,
        special_tokens=special_tokens,
        desired_num_chunks=4  # 使用8个进程
    )
    end_time = time.time()
    print(f"并行预分词时间: {end_time - start_time}秒")
    
    # 训练BPE
    start_time = time.time()
    vocab, merges = train_bpe(
        vocab_size=500,
        special_tokens=special_tokens,
        pre_token_freq=pre_token_freq  # 传入预计算的频率
    )
    end_time = time.time()
    print(f"BPE训练时间: {end_time - start_time}秒")
    
    # 停止性能分析器
    profiler.disable()
    
    # 创建性能统计对象
    stats = pstats.Stats(profiler)
    
    # 打印性能统计信息
    print("\n=== 性能分析结果 ===")
    print("按累计时间排序的前20个函数:")
    stats.sort_stats('cumulative').print_stats(20)
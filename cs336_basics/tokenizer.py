import time
from typing import Dict, Iterable, List, Tuple

from cs336_basics import train_bpe
from cs336_basics.train_bpe import parallel_preprocess_from_file, pre_tokenize, pre_tokenize_iter
from tests.test_tokenizer import MERGES_PATH, VOCAB_PATH, get_tokenizer_from_vocab_merges_path

class BPETokenizer:
    def __init__(self, vocab:Dict[int, bytes], merges:List[Tuple[bytes, bytes]], special_tokens:List[str] = None) -> None:
        """
        初始化BPE分词器
        """
        self.token_vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.token_to_id: Dict[bytes,int] = {token: idx for idx, token in self.token_vocab.items()}

        self.word_to_ids: Dict[bytes, List[int]] = {} # 缓存已经计算过的词的对应id序列

    def words_to_ids(self, word: bytes) -> List[int]:
        """
        将一个word根据词表不断合并，得到其token ID序列
        """
        token_ids = []
        # 将每个字节作为独立的bytes对象
        bytes_list = [bytes([b]) for b in word]  # 将每个字节作为单独的bytes对象

        while len(bytes_list) > 1:
            # 一轮中可能同时满足多个合并规则，选择index最小的合并规则进行合并
            min_rule_idx = None
            min_merge_pos = None
            
            # 遍历当前字节列表中所有可能的合并规则
            for i, pair in enumerate(zip(bytes_list[:-1], bytes_list[1:])):
                idx = self.token_to_id.get(pair[0] + pair[1])
                if (idx is not None) and ((min_rule_idx is None) or (idx < min_rule_idx)):
                    # 找到一个更小的合并规则，更新最小index和位置
                    min_rule_idx = idx
                    min_merge_pos = i
            
            if min_rule_idx is None:
                # 没有可合并的规则
                break
            
            # 执行合并
            # 例如：bytes_list = [b'a', b'b', b'c', b'd']
            # 合并规则：b'b',b'c' -> b'bc'
            # 合并后：bytes_list = [b'a', b'bc', b'd']
            bytes_list[min_merge_pos:min_merge_pos + 2] = [bytes_list[min_merge_pos] + bytes_list[min_merge_pos + 1]]
        
        # 出循环说明已经合并完成，开始翻译为ids
        for part in bytes_list:
            try:
                id = self.token_to_id[part]
                token_ids.append(id)
            except KeyError:
                # 如果没有找到对应的ID，可能是未训练的token,暂时不处理
                print(f"Token {part} 不在词表中.")
                pass
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        对可迭代对象（例如文件句柄）中的每个文本进行编码，每次调用返回一个token ID
        """
        words_iter = pre_tokenize_iter(iterable, self.special_tokens)
        for word in words_iter:
            if word in self.token_to_id:
                # 如果是特殊token/其他单token，直接返回对应的ID
                yield self.token_to_id[word]
            elif word in self.word_to_ids:
                # 如果已经计算过这个词，直接使用缓存
                yield from self.word_to_ids[word]
            else:
                # 计算该词对应token ID序列
                token_ids = self.words_to_ids(word)
                self.word_to_ids[word] = token_ids
                yield from token_ids

    def encode(self, text:str) -> List[int]:
        """
        将文本编码为BPE token ID列表
        """
        # 预分词，把str转为list[bytes]
        words = pre_tokenize(text, self.special_tokens) # word是bytes
        ids = []
        for word in words:
            if word in self.token_to_id:
                # 如果是特殊token/其他单token，直接返回对应的ID
                ids.append(self.token_to_id[word])
            elif word in self.word_to_ids:
                # 如果已经计算过这个词，直接使用缓存
                ids.extend(self.word_to_ids[word])
            else:
                # 计算该词对应token ID序列
                token_ids = self.words_to_ids(word)
                self.word_to_ids[word] = token_ids  # 缓存结果
                ids.extend(token_ids)
        return ids

    def decode(self, ids: Iterable[int], end_token_id: int = None) -> str:
        """
        将BPE token ID列表解码为文本
        """
        text_bytes = b""
        for id in ids:
            if id in self.token_vocab:
                text_bytes += self.token_vocab[id]
            else:
                print(f"Warning: ID {id} not found in vocabulary.")
                continue

            if (end_token_id is not None) and (id == end_token_id):
                break
            
        return text_bytes.decode('utf-8', errors='ignore')

if __name__ == "__main__":
    # 示例：使用并行预分词训练BPE
    file_path = "data/TinyStoriesV2-GPT4-valid.txt"  # 数据集文件路径
    special_tokens = ["<|endoftext|>"]
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

    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    print(tokenizer.encode("Hello, world! <|endoftext|> This is a test."))
    print(tokenizer.decode(tokenizer.encode("Hello, world! <|endoftext|> This is a test.")))
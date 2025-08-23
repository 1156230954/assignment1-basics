# CS336

## BPE Tokenizer

### Pre-tokenization

并行执行预分词效率如下

| file                         | thread | cost time |
| ---------------------------- | ------ | --------- |
| TinyStoriesV2-GPT4-valid.txt | 1      | 3.11s     |
| TinyStoriesV2-GPT4-valid.txt | 4      | 2.32s     |
| TinyStoriesV2-GPT4-valid.txt | 8      | 3.2s      |
| TinyStoriesV2-GPT4-train.txt | 1      | 254s      |
| TinyStoriesV2-GPT4-train.txt | 4      | 69s       |
| TinyStoriesV2-GPT4-train.txt | 8      | 45s       |

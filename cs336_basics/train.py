# train.py
from tokenizer import train_bpe
import json

def main():
    # 设置输入文件路径和词汇表大小
    input_path = 'data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']

    # 调用train_bpe函数
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # 将结果序列化到磁盘
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f)

    with open('merges.json', 'w', encoding='utf-8') as f:
        json.dump(merges, f)

    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

if __name__ == "__main__":
    main()
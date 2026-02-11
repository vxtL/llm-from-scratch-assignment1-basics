import numpy as np
import json
from tokenizer import get_tokenizer
import os
from array import array

def text_to_numpy_streaming(input_txt_path, output_npy_path, vocab_json, merges_json,
                            batch_lines=1000):
    """
    流式处理大文本，避免OOM
    """
    print(f"加载tokenizer...")
    with open(vocab_json, 'r', encoding='utf-8') as f:
        vocab = {int(k): v.encode('utf-8') for k, v in json.load(f).items()}
    with open(merges_json, 'r', encoding='utf-8') as f:
        merges = [tuple(pair) for pair in json.load(f)]
    
    tokenizer = get_tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    
    print(f"流式处理: {input_txt_path}")
    
    # 用array模块紧凑存储uint16，比list省内存
    token_buffer = array('H')  # 'H' = unsigned short (2 bytes)
    total_lines = 0
    
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        batch = []
        
        for line in f:
            batch.append(line.strip())
            total_lines += 1
            
            if len(batch) >= batch_lines:
                # 批量编码，用endoftext分隔
                text = "<|endoftext|>\n".join(batch) + "<|endoftext|>\n"
                tokens = tokenizer.encode(text)
                token_buffer.extend(tokens)
                batch = []
                
                if total_lines % (batch_lines * 100) == 0:
                    print(f"  已处理 {total_lines:,} 行, {len(token_buffer):,} tokens")
        
        # 处理剩余
        if batch:
            text = "<|endoftext|>\n".join(batch) + "<|endoftext|>\n"
            tokens = tokenizer.encode(text)
            token_buffer.extend(tokens)
    
    # 转为numpy保存
    print(f"保存 {len(token_buffer):,} tokens...")
    token_array = np.frombuffer(token_buffer, dtype=np.uint16).copy()
    np.save(output_npy_path, token_array)
    
    print(f"完成: {output_npy_path}")
    return len(token_array)


if __name__ == "__main__":
    VOCAB_PATH = "./vocab_clean.json"
    MERGES_PATH = "./merges_clean.json"
    
    train_tokens = text_to_numpy_streaming(
        input_txt_path="./datasets/data/owt_train.txt",
        output_npy_path="./datasets/npy/owt_train.npy",
        vocab_json=VOCAB_PATH,
        merges_json=MERGES_PATH
    )
    
    val_tokens = text_to_numpy_streaming(
        input_txt_path="./datasets/data/owt_valid.txt",
        output_npy_path="./datasets/npy/owt_val.npy",
        vocab_json=VOCAB_PATH,
        merges_json=MERGES_PATH
    )
    
    print(f"\n数据准备完成！")
    print(f"训练集: {train_tokens:,} tokens")
    print(f"验证集: {val_tokens:,} tokens")
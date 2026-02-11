import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random
import os
import base64
import binascii

# 假设你的tokenizer.py在当前目录
from tokenizer import Tokenizer  # 取消注释这行

def infer_vocab_format(vocab_json: dict) -> str:
    """自动推断vocab文件格式类型"""
    sample_items = list(vocab_json.items())[:10]
    key_types = set(type(k) for k, _ in sample_items)
    value_types = set(type(v) for k, v in sample_items if not str(k).startswith('<|'))
    
    # 格式1: {"0": "<|endoftext|>", "1": "\u0000"} ID->字符串
    if key_types == {str} and all(k.isdigit() for k, _ in sample_items):
        return "id_to_string"
    
    # 格式2: {"<|endoftext|>": 0, "!": 10112} 字符串->ID (Hugging Face)
    if key_types == {str} and (value_types == {int} or 0 in [v for _, v in sample_items]):
        return "string_to_id"
    
    raise ValueError(f"无法识别的vocab格式: keys={key_types}")

def load_custom_tokenizer(vocab_path: str, merges_path: str, special_tokens: List[str] = None):
    """加载自定义BPE分词器 - 支持任意混合格式"""
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(f"Tokenizer files not found: {vocab_path}, {merges_path}")
    
    try:
        # ========== 加载词汇表 ==========
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        
        format_type = infer_vocab_format(vocab_json)
        print(f"  Detected format: {format_type}")
        
        vocab = {}
        
        if format_type == "id_to_string":
            # tinystories格式: {"0": "<|endoftext|>", "1": "\u0000"}
            for k, v in vocab_json.items():
                try:
                    key_int = int(k)
                    # 处理Unicode转义字符串如 "\u0000"
                    if isinstance(v, str):
                        vocab[key_int] = v.encode('utf-8')
                    elif isinstance(v, int):
                        vocab[key_int] = bytes([v])
                except:
                    continue
        
        elif format_type == "string_to_id":
            # openwebtext格式: {"<|endoftext|>": 0, "!": 10112}
            # 反向映射: ID -> token bytes
            reverse_vocab = {}
            for token_str, token_id in vocab_json.items():
                if isinstance(token_id, int) and isinstance(token_str, str):
                    reverse_vocab[token_id] = token_str.encode('utf-8')
            
            # 按ID排序
            vocab = {k: reverse_vocab[k] for k in sorted(reverse_vocab.keys())}
        
        # ========== 加载合并规则 ==========
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)
        
        merges = []
        
        for pair in merges_json:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            
            # 跳过元数据行
            if pair[0] in ["#version:", "", "#version"] or pair[1] in ["0.2", ""]:
                continue
            
            try:
                first = _decode_merge_element(pair[0])
                second = _decode_merge_element(pair[1])
                merges.append((first, second))
            except:
                continue
        
        print(f"  Loaded vocab size: {len(vocab)}, merges: {len(merges)}")
        return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
        
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

def _decode_merge_element(element):
    """解码合并规则中的单个元素"""
    if isinstance(element, list):
        return bytes(element)
    elif isinstance(element, int):
        return bytes([element]) if 0 <= element <= 255 else b''
    elif isinstance(element, str):
        if element == "":
            return b""
        try:
            if len(element) % 4 == 0:
                return base64.b64decode(element, validate=True)
        except:
            pass
        return element.encode('utf-8', errors='replace')
    return b""

def sample_documents(dataset_path: str, num_samples: int = 10) -> List[str]:
    """从数据集中随机采样文档"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    documents = []
    path = Path(dataset_path)
    
    if path.suffix == '.jsonl':
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'text' in data and data['text'].strip():
                        documents.append(data['text'].strip())
                except json.JSONDecodeError:
                    continue
    else:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
    
    if len(documents) < num_samples:
        print(f"Warning: Only {len(documents)} documents available")
        return documents
    
    return random.sample(documents, num_samples)

def calculate_compression_stats(text: str, tokenizer) -> Tuple[float, int, int]:
    """计算压缩统计信息"""
    if not text:
        return 0.0, 0, 0
    
    byte_size = len(text.encode('utf-8'))
    token_ids = tokenizer.encode(text)
    token_count = len(token_ids)
    compression_ratio = byte_size / token_count if token_count > 0 else 0.0
    
    return compression_ratio, token_count, byte_size

def main():
    # ==================== 配置区 ====================
    
    # 数据集路径
    DATA_PATHS = {
        "tinystories": "data/TinyStoriesV2-GPT4-valid.txt",
        "openwebtext": "data/owt_valid.txt"
    }
    
    # 分词器路径（你的自定义格式）
    TOKENIZER_PATHS = {
        "tinystories": {
            "vocab": "vocab.json",
            "merges": "merges.json",
            "special_tokens": ["<|endoftext|>"]  # 根据你的训练配置调整
        },
        "openwebtext": {
            "vocab": "my_bpe_tokenizer-no-g-vocab.json",
            "merges": "my_bpe_tokenizer-no-g-merges.json",
            "special_tokens": ["<|endoftext|>"]  # 根据你的训练配置调整
        }
    }
    
    NUM_SAMPLES = 100
    
    # ====================================================================
    
    print("="*70)
    print("BPE Tokenizer Compression Ratio Experiment")
    print("="*70)
    
    # Step 1: 加载分词器
    print("\n【Step 1】Loading tokenizers...")
    tokenizers = {}
    for name, paths in TOKENIZER_PATHS.items():
        try:
            print(f"  Loading {name}...")
            tokenizer = load_custom_tokenizer(
                paths["vocab"], 
                paths["merges"], 
                paths.get("special_tokens", [])
            )
            tokenizers[name] = tokenizer
            vocab_size = len(tokenizer.vocab)
            print(f"  ✓ {name} tokenizer loaded (vocab size: {vocab_size})")
        except Exception as e:
            print(f"  ✗ Failed to load {name} tokenizer: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Step 2: 采样文档
    print("\n【Step 2】Sampling documents...")
    samples = {}
    for name, path in DATA_PATHS.items():
        try:
            docs = sample_documents(path, NUM_SAMPLES)
            samples[name] = docs
            print(f"  ✓ {name}: {len(docs)} documents sampled")
        except Exception as e:
            print(f"  ✗ Failed to sample {name}: {e}")
            return
    
    # Step 3: 计算压缩比
    print("\n【Step 3】Calculating compression ratios...")
    print("-"*70)
    
    results = {}
    
    # 实验1: TinyStories分词器 + TinyStories数据
    print("\nExperiment 1: TinyStories Tokenizer → TinyStories Data")
    ts_ratios, total_ts_bytes, total_ts_tokens = [], 0, 0
    
    for i, doc in enumerate(samples["tinystories"], 1):
        ratio, tokens, bytes_ = calculate_compression_stats(doc, tokenizers["tinystories"])
        ts_ratios.append(ratio)
        total_ts_bytes += bytes_
        total_ts_tokens += tokens
        print(f"  Doc {i:2d} | {bytes_:5d} bytes → {tokens:4d} tokens | Ratio: {ratio:6.2f} bytes/token")
    
    results["ts_on_ts"] = {
        "avg_ratio": np.mean(ts_ratios),
        "overall_ratio": total_ts_bytes / total_ts_tokens if total_ts_tokens > 0 else 0,
        "total_bytes": total_ts_bytes,
        "total_tokens": total_ts_tokens
    }
    
    # 实验2: OpenWebText分词器 + OpenWebText数据
    print("\nExperiment 2: OpenWebText Tokenizer → OpenWebText Data")
    owt_ratios, total_owt_bytes, total_owt_tokens = [], 0, 0
    
    for i, doc in enumerate(samples["openwebtext"], 1):
        ratio, tokens, bytes_ = calculate_compression_stats(doc, tokenizers["openwebtext"])
        owt_ratios.append(ratio)
        total_owt_bytes += bytes_
        total_owt_tokens += tokens
        print(f"  Doc {i:2d} | {bytes_:5d} bytes → {tokens:4d} tokens | Ratio: {ratio:6.2f} bytes/token")
    
    results["owt_on_owt"] = {
        "avg_ratio": np.mean(owt_ratios),
        "overall_ratio": total_owt_bytes / total_owt_tokens if total_owt_tokens > 0 else 0,
        "total_bytes": total_owt_bytes,
        "total_tokens": total_owt_tokens
    }
    
    # 打印总结
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nTinyStories Tokenizer:")
    print(f"  Average Ratio: {results['ts_on_ts']['avg_ratio']:.2f} bytes/token")
    print(f"  Overall Ratio: {results['ts_on_ts']['overall_ratio']:.2f} bytes/token")
    print(f"  Total: {results['ts_on_ts']['total_bytes']} bytes → {results['ts_on_ts']['total_tokens']} tokens")
    
    print(f"\nOpenWebText Tokenizer:")
    print(f"  Average Ratio: {results['owt_on_owt']['avg_ratio']:.2f} bytes/token")
    print(f"  Overall Ratio: {results['owt_on_owt']['overall_ratio']:.2f} bytes/token")
    print(f"  Total: {results['owt_on_owt']['total_bytes']} bytes → {results['owt_on_owt']['total_tokens']} tokens")
    
    # 计算相对效率
    ts_eff = results['ts_on_ts']['overall_ratio']
    owt_eff = results['owt_on_owt']['overall_ratio']
    improvement = (ts_eff - owt_eff) / ts_eff * 100 if ts_eff > 0 else 0
    if improvement > 0:
        print(f"\nOpenWebText tokenizer is {improvement:.1f}% more bytes/token efficient")
    else:
        print(f"\nTinyStories tokenizer is {abs(improvement):.1f}% more bytes/token efficient")
    
    # 保存结果
    output_file = "compression_ratio_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "tiny_stories": {
                    "overall_compression_ratio": round(results['ts_on_ts']['overall_ratio'], 2),
                    "total_bytes": results['ts_on_ts']['total_bytes'],
                    "total_tokens": results['ts_on_ts']['total_tokens']
                },
                "openwebtext": {
                    "overall_compression_ratio": round(results['owt_on_owt']['overall_ratio'], 2),
                    "total_bytes": results['owt_on_owt']['total_bytes'],
                    "total_tokens": results['owt_on_owt']['total_tokens']
                },
                "efficiency_improvement_percent": round(abs(improvement), 1),
                "more_efficient_tokenizer": "openwebtext" if improvement > 0 else "tinystories"
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
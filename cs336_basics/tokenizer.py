# cs336_basics/tokenizer.py
from __future__ import annotations
import collections
from pathlib import Path
import regex as re
from typing import Any, Iterable, Iterator
import heapq
from typing import Tuple, Dict, List
from collections import defaultdict, Counter
# GPT-2 标准预分词正则
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练 BPE 分词器。
    """
    special_tokens = special_tokens or []
    
    # 1. 初始化词表：先放 special tokens，再放 256 个字节
    vocab: dict[int, bytes] = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # 2. 读取文件并预处理
    text = Path(input_path).read_text(encoding="utf-8")
    
    # 修复点 1：按长度降序排列 special tokens 以确保最长匹配
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    if sorted_special_tokens:
        split_pattern = "|".join(re.escape(st) for st in sorted_special_tokens)
        parts = re.split(f"({split_pattern})", text)
    else:
        parts = [text]
        
    special_token_set = set(special_tokens)
    word_counts = collections.Counter()
    for part in parts:
        if part in special_token_set or not part:
            continue
        # 对非 special token 区域进行正则分词
        for match in PAT.finditer(part):
            word_counts[match.group(0).encode("utf-8")] += 1

    # 初始化：将每个单词表示为 token ID 序列
    # byte 对应的 ID 是 len(special_tokens) + byte_value
    byte_to_id = {bytes([i]): len(special_tokens) + i for i in range(256)}
    unique_words = {w: [byte_to_id[bytes([b])] for b in w] for w in word_counts}
    
    merges: list[tuple[bytes, bytes]] = []
    
    for _ in range(num_merges):
        # 3. 统计当前所有相邻 pair 的频率
        pair_counts = collections.defaultdict(int)
        for word_bytes, ids in unique_words.items():
            count = word_counts[word_bytes]
            for i in range(len(ids) - 1):
                pair_counts[(ids[i], ids[i+1])] += count
        
        if not pair_counts:
            break
            
        # 4. 选择最佳 pair：最高频，频率相同时选字节序列字典序最大的（Tie-breaking）
        best_pair = max(
            pair_counts.keys(),
            key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]])
        )
        
        # 5. 更新词表和 merges
        id1, id2 = best_pair
        new_id = len(vocab)
        new_bytes = vocab[id1] + vocab[id2]
        vocab[new_id] = new_bytes
        merges.append((vocab[id1], vocab[id2]))
        
        # 6. 更新所有唯一单词的 ID 序列
        for word_bytes in unique_words:
            ids = unique_words[word_bytes]
            if len(ids) < 2:
                continue
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == id1 and ids[i+1] == id2:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            unique_words[word_bytes] = new_ids
            
    return vocab, merges

class Tokenizer:
    """
    BPE Tokenizer 类，负责编码和解码。
    """
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.special_tokens = special_tokens or []
        
        # 修复点1：构建完整的byte_to_id映射，与train_bpe保持一致
        self.byte_to_id = {}
        
        # 1. 首先添加special tokens
        for i, token in enumerate(self.special_tokens):
            token_bytes = token.encode("utf-8")
            if token_bytes in vocab.values():
                # 如果vocab中已存在，找到其ID
                for tid, tbytes in vocab.items():
                    if tbytes == token_bytes:
                        self.byte_to_id[token_bytes] = tid
                        break
            else:
                # 否则使用训练时的约定ID
                self.byte_to_id[token_bytes] = i
        
        # 2. 确保所有256个字节值都有正确映射
        start_byte_id = len(self.special_tokens)
        for i in range(256):
            byte_val = bytes([i])
            # 如果vocab中已存在该字节，使用其ID；否则使用标准偏移
            found = False
            for tid, tbytes in vocab.items():
                if tbytes == byte_val:
                    self.byte_to_id[byte_val] = tid
                    found = True
                    break
            if not found:
                # 使用训练时的标准偏移
                self.byte_to_id[byte_val] = start_byte_id + i
        
        # 3. 添加BPE合并产生的token
        for token_id, token_bytes in vocab.items():
            # 只添加非字节值的token（避免覆盖）
            if len(token_bytes) > 1 or (len(token_bytes) == 1 and token_bytes[0] >= 256):
                self.byte_to_id[token_bytes] = token_id
        
        # 记录 merge 的先后顺序
        self.merges = {pair: i for i, pair in enumerate(merges)}

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        # 按长度降序排列 special tokens 以确保最长匹配
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        if sorted_special_tokens:
            split_pattern = "|".join(re.escape(st) for st in sorted_special_tokens)
            parts = re.split(f"({split_pattern})", text)
        else:
            parts = [text]

        token_ids = []
        special_token_set = set(self.special_tokens)

        for part in parts:
            if not part:
                continue
            if part in special_token_set:
                token_ids.append(self.byte_to_id[part.encode("utf-8")])
            else:
                # 按照 GPT-2 逻辑对每个预分词单元进行合并
                for match in PAT.finditer(part):
                    word_bytes = match.group(0).encode("utf-8")
                    
                    # 修复点2：安全地获取每个字节的ID
                    word_ids = []
                    for b in word_bytes:
                        byte_val = bytes([b])
                        # 使用get并设置默认值防止KeyError
                        word_ids.append(self.byte_to_id.get(byte_val, len(self.special_tokens) + b))
                    
                    # 堆合并逻辑...
                    heap = []
                    for i in range(len(word_ids) - 1):
                        if word_ids[i] in self.vocab and word_ids[i + 1] in self.vocab:
                            pair = (self.vocab[word_ids[i]], self.vocab[word_ids[i + 1]])
                            rank = self.merges.get(pair, float('inf'))
                            heapq.heappush(heap, (rank, i))

                    while heap:
                        rank, best_pair_idx = heapq.heappop(heap)
                        if rank == float('inf') or best_pair_idx >= len(word_ids) - 1:
                            break

                        id1, id2 = word_ids[best_pair_idx], word_ids[best_pair_idx + 1]
                        new_token = self.vocab.get(id1, b"") + self.vocab.get(id2, b"")
                        new_id = self.byte_to_id.get(new_token, None)
                        if new_id is None:
                            break

                        word_ids = word_ids[:best_pair_idx] + [new_id] + word_ids[best_pair_idx + 2:]

                        new_heap = []
                        for i in range(len(word_ids) - 1):
                            if word_ids[i] in self.vocab and word_ids[i + 1] in self.vocab:
                                pair = (self.vocab[word_ids[i]], self.vocab[word_ids[i + 1]])
                                rank = self.merges.get(pair, float('inf'))
                                heapq.heappush(new_heap, (rank, i))
                        heap = new_heap

                    token_ids.extend(word_ids)

        return token_ids

    def decode(self, ids: list[int]) -> str:
        """
        解码 ID 序列，无效 UTF-8 字节将被替换。
        """
        byte_seq = b"".join(self.vocab[idx] for idx in ids)
        return byte_seq.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return Tokenizer(vocab, merges, special_tokens)
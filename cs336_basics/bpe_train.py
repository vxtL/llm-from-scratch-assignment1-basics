from tokenizer import train_bpe
import json
import time
import psutil
import os
import sys
from multiprocessing import Pool, Value, Lock

# 全局变量，用于实时进度输出
progress_counter = Value('i', 0)  # 初始化计数器
progress_lock = Lock()  # 初始化锁

def train_chunk(chunk_path, vocab_size, special_tokens, chunk_index):
    """
    训练单个数据块的 BPE 分词器。
    """
    print(f"开始训练块: {chunk_index + 1} - {chunk_path}")
    vocab, merges = train_bpe(chunk_path, vocab_size, special_tokens)
    print(f"完成训练块: {chunk_index + 1} - {chunk_path}")
    return vocab, merges

def split_file(input_path, chunk_size_mb, output_dir):
    """
    将大文件分割成多个小块。
    """
    chunk_size = chunk_size_mb * 1024 * 1024
    with open(input_path, 'r', encoding='utf-8') as f:
        chunk_number = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunk_path = os.path.join(output_dir, f'chunk_{chunk_number}.txt')
            with open(chunk_path, 'w', encoding='utf-8') as chunk_file:
                chunk_file.write(chunk)
            chunk_number += 1
    return chunk_number

def merge_results(results):
    """
    合并多个数据块的训练结果。
    """
    combined_vocab = {}
    combined_merges = []
    for vocab, merges in results:
        combined_vocab.update(vocab)
        combined_merges.extend(merges)
    return combined_vocab, combined_merges

def update_progress(chunk_index, total_chunks):
    """
    更新进度并打印实时进度。
    """
    with progress_lock:
        progress_counter.value += 1
        print(f"[{progress_counter.value}/{total_chunks}] 数据块已完成 - 完成第 {chunk_index} 块", end="\r")

def main():
    # ==================== 配置参数 ====================
    USE_SAMPLE_MODE = True  # 设置为 True 启用采样模式（测试），False 为完整训练
    SAMPLE_LINES = 5000    # 采样行数（建议 1000-10000 行，约 5-50MB）
    input_path = 'owt_train_with_special_token.txt'  # 完整数据路径
    sample_path = 'owt_train-sample.txt'  # 采样数据输出路径（自动创建）
    vocab_size = 32000  # 修改词汇表大小为 32,000
    special_tokens = ['<|endoftext|>']  # 保留特殊标记
    chunk_size_mb = 1  # 每个数据块的大小（MB）
    num_processes = 208  # 使用的进程数
    # =================================================

    # 步骤1：检查数据文件
    if not os.path.exists(input_path):
        print(f"❌ 错误: 文件不存在: {input_path}")
        print("\n可用文件列表:")
        os.system("ls -lh data/")
        sys.exit(1)

    # 步骤2：如果使用采样模式，创建小文件
    if USE_SAMPLE_MODE:
        print(f"🧪 采样模式已启用: 读取前 {SAMPLE_LINES} 行")
        print(f"正在创建采样文件: {sample_path}...")
        
        with open(input_path, 'r', encoding='utf-8') as f_in:
            # 只读取前 SAMPLE_LINES 行
            sample_lines = [next(f_in) for _ in range(SAMPLE_LINES)]
        
        with open(sample_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(sample_lines)
        
        time.sleep(1)  # 等待1秒，确保文件完全写入
        # 使用采样文件作为输入
        actual_input_path = sample_path
        print(f"✅ 采样文件创建完成，大小: {os.path.getsize(sample_path) / 1024 / 1024:.2f} MB\n")
    else:
        print("📦 完整训练模式")
        actual_input_path = input_path

    # 步骤3：分割文件
    output_dir = 'chunksss'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("正在分割文件...")
    num_chunks = split_file(actual_input_path, chunk_size_mb, output_dir)
    print(f"文件已分割成 {num_chunks} 个块，每个块约 {chunk_size_mb} MB")

    # 步骤4：多进程训练
    print("开始多进程训练...")
    chunk_paths = [os.path.join(output_dir, f'chunk_{i}.txt') for i in range(num_chunks)]

    # 使用 Pool.map_async 并添加回调函数更新进度
    with Pool(processes=num_processes) as pool:
        results = pool.starmap_async(train_chunk, [(path, vocab_size, special_tokens, i) for i, path in enumerate(chunk_paths)], callback=lambda _: update_progress(progress_counter.value, num_chunks))
        results.get()  # 等待所有任务完成

    # 步骤5：合并结果
    combined_vocab, combined_merges = merge_results(results.get())

    # 步骤6：保存结果
    print("\n正在保存结果...")
    with open('vocab.json', 'w', encoding='utf-8') as f:
        vocab_serializable = {
            str(k): v.decode('utf-8', errors='replace') for k, v in combined_vocab.items()
        }
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

    with open('merges.json', 'w', encoding='utf-8') as f:
        merges_serializable = [
            (p[0].decode('utf-8', errors='replace'), 
             p[1].decode('utf-8', errors='replace'))
            for p in combined_merges
        ]
        json.dump(merges_serializable, f, ensure_ascii=False, indent=2)

    print("训练完成！")
    print(f"文件已保存:")
    print(f"  - vocab.json ({os.path.getsize('vocab.json') / 1024:.2f} KB)")
    print(f"  - merges.json ({os.path.getsize('merges.json') / 1024:.2f} KB)")

    # 步骤7：清理采样文件和数据块
    if USE_SAMPLE_MODE and os.path.exists(sample_path):
        os.remove(sample_path)
        print(f"\n🗑️ 已删除临时采样文件: {sample_path}")
    for chunk_path in chunk_paths:
        os.remove(chunk_path)
    print(f"🗑️ 已删除临时数据块文件")

if __name__ == "__main__":
    main()
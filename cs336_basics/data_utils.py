import numpy as np
import torch

def get_batch(x, batch_size, context_length, device):
    """
    实现数据加载函数，支持内存映射数组
    
    参数:
        x: 包含标记 ID 的 NumPy 数组或内存映射数组（np.memmap）
           支持通过 np.load(..., mmap_mode='r') 或 np.memmap() 加载的大文件
        batch_size: 每个批次的大小
        context_length: 上下文长度
        device: PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'）
    
    返回:
        输入序列和下一个标记目标的张量对，形状为 (batch_size, context_length)
    """
    # 计算最大可能的起始位置（确保不会越界）
    max_start_idx = len(x) - context_length - 1
    
    # 随机选择 batch_size 个起始位置
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # 初始化输入和目标张量
    inputs = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    targets = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    
    # 填充批次（内存映射数组支持高效切片，不会加载整个文件）
    for i, start_idx in enumerate(start_indices):
        # 输入：从 start_idx 开始，长度为 context_length
        inputs[i] = torch.tensor(x[start_idx:start_idx + context_length], device=device)
        
        # 目标：从 start_idx+1 开始，长度为 context_length（向右偏移一位）
        targets[i] = torch.tensor(x[start_idx + 1:start_idx + context_length + 1], device=device)
    
    return inputs, targets



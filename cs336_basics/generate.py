# decode.py
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np

# === 验证导入 ===
import os
import sys

# 检查必要的模块
required_files = {
    "model.py": ["softmax", "TransformerLM"],
}

missing_modules = []
for filename, functions in required_files.items():
    if not os.path.exists(filename):
        missing_modules.append(f"{filename}")
        continue
    
    try:
        module_name = filename.replace('.py', '')
        module = __import__(module_name)
        for func_name in functions:
            if not hasattr(module, func_name):
                missing_modules.append(f"{filename}:{func_name}")
    except ImportError as e:
        missing_modules.append(f"{filename} (ImportError: {e})")

if missing_modules:
    print("错误：以下必需模块或函数缺失：")
    for m in missing_modules:
        print(f"  - {m}")
    sys.exit(1)

# === 正式导入 ===
from model import softmax, TransformerLM

def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    应用温度缩放调整概率分布的平滑度
    
    参数:
        logits: 模型的原始logits输出
        temperature: 温度参数
                   - temperature → 0: 分布更尖锐（更确定）
                   - temperature → ∞: 分布更均匀（更随机）
    
    返回:
        缩放后的logits
    """
    if temperature <= 0:
        raise ValueError("温度必须为正数")
    return logits / temperature


def top_p_sampling(probs: torch.Tensor, top_p: float, filter_value: float = 0.0) -> torch.Tensor:
    """
    核采样（nucleus sampling）：只保留概率最高的token，直到累积概率达到top_p
    
    参数:
        probs: 概率分布 (形状: [vocab_size])
        top_p: 累积概率阈值 (0 < top_p <= 1)
        filter_value: 被过滤token的值设为多少
    
    返回:
        过滤后的概率分布
    """
    if not (0 < top_p <= 1):
        raise ValueError("top_p必须在(0, 1]范围内")
    
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过top_p的位置
    # 保留那些累积概率 <= top_p的token
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # 至少保留一个token（确保能采样）
    sorted_indices_to_remove[0] = False
    
    # 将需要移除的token转换回原始顺序
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, 
        index=sorted_indices, 
        src=sorted_indices_to_remove
    )
    
    # 将需要移除的token概率设为filter_value
    probs = probs.clone()
    probs[indices_to_remove] = filter_value
    
    # 重新归一化概率（确保总和为1）
    probs_sum = probs.sum()
    if probs_sum > 0:
        probs = probs / probs_sum
    else:
        # 如果所有token都被过滤，退回到均匀分布
        probs = torch.ones_like(probs) / probs.numel()
    
    return probs


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    device: str = "cpu"
) -> Tuple[int, torch.Tensor]:
    """
    从logits采样下一个token
    
    参数:
        logits: 模型的logits输出 (形状: [vocab_size])
        temperature: 温度参数
        top_p: 核采样阈值 (None表示不使用)
        device: 设备
    
    返回:
        (采样得到的token_id, 应用的最终概率分布)
    """
    # 应用温度缩放
    scaled_logits = temperature_scaling(logits, temperature)
    
    # 使用自定义softmax转换为概率
    probs = softmax(scaled_logits, dim=-1)
    
    # 应用top-p采样（如果指定）
    if top_p is not None:
        probs = top_p_sampling(probs, top_p)
    
    # 从概率分布中采样
    token_id = torch.multinomial(probs, num_samples=1).item()
    
    return token_id, probs


def decode(
    model: TransformerLM,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    end_token: str = "<|endoftext|>",
    device: str = "cpu"
) -> Tuple[str, List[int]]:
    """
    从语言模型生成文本
    
    参数:
        model: Transformer语言模型
        tokenizer: 分词器
        prompt: 提示文本
        max_tokens: 最大生成token数
        temperature: 温度参数
        top_p: 核采样阈值 (None表示不使用)
        end_token: 结束token字符串
        device: 设备
    
    返回:
        (生成的文本, 生成的token_id列表)
    """
    model.eval()
    
    # 编码提示词
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [tokenizer.encode(" ")[0]]  # 空提示词处理
    
    generated_ids = input_ids.copy()
    
    # 获取结束token的ID
    end_token_id = tokenizer.encode(end_token)[0] if end_token else None
    
    with torch.no_grad():
        for step in range(max_tokens):
            # 准备输入张量
            input_tensor = torch.tensor([generated_ids], device=device)
            
            # 前向传播
            logits = model(input_tensor)
            
            # 获取最后一个位置的logits
            last_logits = logits[0, -1, :]  # [vocab_size]
            
            # 采样下一个token
            next_token_id, probs = sample_next_token(
                last_logits, 
                temperature=temperature, 
                top_p=top_p,
                device=device
            )
            
            # 添加到生成序列
            generated_ids.append(next_token_id)
            
            # 检查是否生成结束token
            if end_token_id is not None and next_token_id == end_token_id:
                break
    
    # 解码为文本
    generated_text = tokenizer.decode(generated_ids[len(input_ids):])
    full_text = tokenizer.decode(generated_ids)
    
    model.train()
    
    return full_text, generated_ids[len(input_ids):]


# === 适配器函数（用于测试框架） ===
def run_decode(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    device: str
) -> Tuple[str, List[int]]:
    """
    适配器函数：连接测试框架
    
    参数:
        model: 模型
        tokenizer: 分词器
        prompt: 提示词
        max_tokens: 最大生成token数
        temperature: 温度
        top_p: top-p采样阈值
        device: 设备
    
    返回:
        (生成的完整文本, 生成的token_id列表)
    """
    return decode(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device
    )


# === 实用函数：批量生成 ===
def generate_batch(
    model: TransformerLM,
    tokenizer: Any,
    prompts: List[str],
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    device: str = "cpu"
) -> List[str]:
    """
    批量生成文本（可选功能，提高效率）
    
    参数:
        prompts: 提示词列表
        max_tokens: 每个提示词的最大生成token数
        temperature: 温度参数
        top_p: 核采样阈值
        device: 设备
    
    返回:
        生成的文本列表
    """
    # 注意：实际批量生成需要处理不同长度的序列，
    # 为了简单起见，这里对每个提示词单独调用decode
    results = []
    for prompt in prompts:
        text, _ = decode(model, tokenizer, prompt, max_tokens, temperature, top_p, device)
        results.append(text)
    return results


# === 示例使用 ===
if __name__ == "__main__":
    # 示例：生成文本
    # 注意：此示例需要已训练的模型和分词器
    """
    # 假设已有 model, tokenizer, device
    prompt = "Once upon a time"
    
    # 贪婪解码（温度=0）
    text, tokens = decode(
        model, tokenizer, prompt, 
        max_tokens=100, 
        temperature=0.0,
        device=device
    )
    print("贪婪解码:", text)
    
    # 随机采样（温度=1.0）
    text, tokens = decode(
        model, tokenizer, prompt, 
        max_tokens=100, 
        temperature=1.0,
        device=device
    )
    print("随机采样:", text)
    
    # 核采样（top_p=0.9）
    text, tokens = decode(
        model, tokenizer, prompt, 
        max_tokens=100, 
        temperature=0.8,
        top_p=0.9,
        device=device
    )
    print("核采样:", text)
    """
    pass
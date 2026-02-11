import argparse
import os
import sys
import time
import logging
from pathlib import Path
import numpy as np
import torch
import torch
import torch.nn.functional as F
import wandb  # 可选的W&B集成

# === 验证导入 ===
# 检查必要的自定义模块是否存在
required_files = {
    "check_points.py": ["save_checkpoint", "load_checkpoint"],
    "data_utils.py": ["get_batch"],
    "model.py": ["TransformerLM", "TransformerBlock", "MultiHeadAttention", "scaled_dot_product_attention"],
    "optimizer.py": ["AdamW", "cosine_schedule_with_warmup", "gradient_clipping"],
    "tokenizer.py": ["train_bpe", "Tokenizer", "get_tokenizer"],
}

missing_modules = []
for filename, functions in required_files.items():
    if not os.path.exists(filename):
        missing_modules.append(f"{filename}")
        continue
    
    # 尝试导入并检查函数是否存在
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
    print("\n请确保所有文件都在当前目录或Python路径中。")
    sys.exit(1)

# === 正式导入 ===
import torch.nn as nn
from check_points import save_checkpoint, load_checkpoint
from data_utils import get_batch
from model import TransformerLM, TransformerBlock, MultiHeadAttention, scaled_dot_product_attention
from optimizer import AdamW, cosine_schedule_with_warmup, gradient_clipping
from tokenizer import train_bpe, Tokenizer, get_tokenizer

def load_dataset_memory_efficient(file_path: str, dtype: np.dtype = np.uint16) -> np.ndarray:
    """
    使用内存映射高效加载大型数据集
    
    参数:
        file_path: .npy 文件路径
        dtype: 数据类型，应与保存时一致
    
    返回:
        内存映射数组，按需加载
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    # 加载内存映射数组（不会一次性加载到内存）
    data = np.load(file_path, mmap_mode='r')
    
    # 验证数据类型
    if data.dtype != dtype:
        print(f"警告：数据类型不匹配。期望 {dtype}，实际 {data.dtype}")
        print("请确保保存和加载时使用相同的 dtype（建议使用 uint16）")
    
    # 验证数据范围（确保token ID在合理范围内）
    if data.max() > 50000:  # 根据你的vocab_size调整
        print(f"警告：发现异常大的token ID: {data.max()}")
    if data.min() < 0:
        raise ValueError(f"发现负的token ID: {data.min()}")
    
    print(f"成功加载内存映射数据集: {file_path}")
    print(f"  数据形状: {data.shape}")
    print(f"  数据类型: {data.dtype}")
    print(f"  数据范围: [{data.min()}, {data.max()}]")
    
    return data


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="训练Transformer语言模型")
    
    # 模型架构参数
    parser.add_argument("--vocab-size", type=int, default=10000,
                        help="词汇表大小")
    parser.add_argument("--context-length", type=int, default=256,
                        help="最大上下文长度")
    parser.add_argument("--d-model", type=int, default=512,
                        help="模型维度")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Transformer层数")
    parser.add_argument("--num-heads", type=int, default=16,
                        help="注意力头数")
    parser.add_argument("--d-ff", type=int, default=1344,
                        help="前馈网络维度")
    parser.add_argument("--rope-theta", type=float, default=10000.0,
                        help="RoPE theta参数")
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--total-tokens", type=int, default=327680000,
                        help="总训练token数")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="初始学习率")
    parser.add_argument("--min-lr", type=float, default=1e-5,
                        help="最小学习率")
    parser.add_argument("--warmup-iters", type=int, default=1000,
                        help="warmup迭代次数")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="权重衰减系数")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="AdamW beta2")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="梯度裁剪阈值")
    
    # 数据参数
    parser.add_argument("--train-data", type=str, required=True,
                        help="训练数据路径 (.npy文件)")
    parser.add_argument("--val-data", type=str, required=True,
                        help="验证数据路径 (.npy文件)")
    parser.add_argument("--tokenizer-vocab", type=str, required=True,
                        help="分词器词汇文件路径")
    parser.add_argument("--tokenizer-merges", type=str, required=True,
                        help="分词器merges文件路径")
    
    # 检查点和日志参数
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="检查点保存目录")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="保存检查点的间隔（迭代次数）")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="日志记录间隔（迭代次数）")
    parser.add_argument("--val-interval", type=int, default=500,
                        help="验证间隔（迭代次数）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练")
    
    # W&B日志参数
    parser.add_argument("--use-wandb", action="store_true",
                        help="使用Weights & Biases记录日志")
    parser.add_argument("--wandb-project", type=str, default="cs336-assignment1",
                        help="W&B项目名称")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B运行名称")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    
    return parser.parse_args()


def calculate_num_iterations(total_tokens: int, batch_size: int, context_length: int) -> int:
    """
    计算总迭代次数
    
    参数:
        total_tokens: 总训练token数
        batch_size: 批次大小
        context_length: 上文长度
    
    返回:
        迭代次数
    """
    tokens_per_batch = batch_size * context_length
    num_iterations = total_tokens // tokens_per_batch
    return num_iterations


def evaluate_model(model: TransformerLM, data: np.ndarray, batch_size: int, 
                   context_length: int, device: str, max_batches: int = 50) -> float:
    """
    评估模型在验证集上的性能
    
    参数:
        model: Transformer模型
        data: 验证数据
        batch_size: 批次大小
        context_length: 上下文长度
        device: 设备
        max_batches: 最大评估批次（避免验证时间过长）
    
    返回:
        平均验证损失
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for _ in range(max_batches):
            try:
                x, y = get_batch(data, batch_size, context_length, device)
                
                # 前向传播
                logits = model(x)
                
                # 计算损失
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1
                )
                
                total_loss += loss.item()
                num_batches += 1
                
            except IndexError:
                # 数据不足，跳出
                break
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def setup_logging(log_dir: str = "./logs"):
    """
    配置日志记录器
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(filename=Path(log_dir / "training.log")),
            logging.StreamHandler()
        ]
    )


def main():
    """
    主训练函数
    """
    args = parse_args()
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    import json  # 添加到文件顶部的import部分
    
    # 加载分词器（JSON格式）
    print("加载分词器...")
    with open(args.tokenizer_vocab, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    with open(args.tokenizer_merges, 'r', encoding='utf-8') as f:
        merges_data = json.load(f)
    
    
    tokenizer = get_tokenizer(vocab_data, merges_data, special_tokens=["<|endoftext|>"])
    
    # 加载数据集（内存映射模式）
    print("加载训练数据...")
    train_data = load_dataset_memory_efficient(args.train_data, dtype=np.uint16)
    
    print("加载验证数据...")
    val_data = load_dataset_memory_efficient(args.val_data, dtype=np.uint16)
    
    # 计算迭代次数
    num_iterations = calculate_num_iterations(args.total_tokens, args.batch_size, args.context_length)
    print(f"总迭代次数: {num_iterations}")
    
    # 构建模型
    print("构建模型...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)
    
    # 构建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    
    # 初始化W&B
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        wandb.watch(model, log="all", log_freq=args.log_interval)
    
    # 从检查点恢复（如果指定）
    start_iteration = 0
    if args.resume:
        print(f"从检查点恢复: {args.resume}")
        start_iteration = load_checkpoint(args.resume, model, optimizer)
        print(f"从迭代 {start_iteration} 开始")
    
    # 训练循环
    print("开始训练...")
    model.train()
    setup_logging(args.checkpoint_dir)  # 设置日志记录器
    start_time = time.time()
    batch_time = start_time
    
    # 最佳验证损失（用于早停）
    best_val_loss = float('inf')
    
    for iteration in range(start_iteration, num_iterations):
        # 1. 获取批次数据
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        
        # 2. 前向传播
        logits = model(x)
        
        # 3. 计算损失
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-1
        )
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 5. 梯度裁剪
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        
        # 6. 学习率调度（预热 + 余弦退火）
        lr = cosine_schedule_with_warmup(
            t=iteration,
            alpha_max=args.lr,
            alpha_min=args.min_lr,
            T_w=args.warmup_iters,
            T_c=num_iterations
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # 7. 优化器步骤
        optimizer.step()
        
        # 8. 日志记录
        if iteration % args.log_interval == 0:
            current_time = time.time()
            elapsed = current_time - start_time
            batches_per_sec = args.log_interval / (current_time - batch_time) if iteration > 0 else 0
            batch_time = current_time
            
            logging.info(f"Iter {iteration:6d}/{num_iterations} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed:.1f}s | "
                  f"Speed: {batches_per_sec:.2f} batches/sec")
            
            if args.use_wandb:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": lr,
                    "iteration": iteration,
                    "batches_per_sec": batches_per_sec,
                    "elapsed_time": elapsed
                })
        
        # 9. 评估
        if iteration % args.val_interval == 0:
            val_loss = evaluate_model(model, val_data, args.batch_size, 
                                      args.context_length, args.device)
            logging.info(f"Iter {iteration:6d} | Validation Loss: {val_loss:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "val_loss": val_loss,
                    "iteration": iteration,
                    "val_ppl": torch.exp(torch.tensor(val_loss)).item()
                })
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.checkpoint_dir, "best_checkpoint.pt")
                save_checkpoint(model, optimizer, iteration, best_path)
                logging.info(f"  → 保存最佳检查点 (val_loss={val_loss:.4f})")
        
        # 10. 保存检查点
        if iteration % args.save_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_iter_{iteration}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            logging.info(f"  → 保存检查点: {checkpoint_path}")
    
    # 训练结束
    logging.info(f"训练完成！总时间: {time.time() - start_time:.1f}秒")
    
    # 保存最终检查点
    final_path = os.path.join(args.checkpoint_dir, "final_checkpoint.pt")
    save_checkpoint(model, optimizer, num_iterations, final_path)
    logging.info(f"最终检查点保存至: {final_path}")
    
    # 关闭W&B
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
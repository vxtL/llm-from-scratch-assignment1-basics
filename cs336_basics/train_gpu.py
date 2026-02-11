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
    "cs336_basics/check_points.py": ["save_checkpoint", "load_checkpoint"],
    "cs336_basics/data_utils.py": ["get_batch"],
    "cs336_basics/model.py": ["TransformerLM", "TransformerBlock", "MultiHeadAttention", "scaled_dot_product_attention"],
    "cs336_basics/optimizer.py": ["AdamW", "cosine_schedule_with_warmup", "gradient_clipping"],
    "cs336_basics/tokenizer.py": ["train_bpe", "Tokenizer", "get_tokenizer"],
}
missing_modules = []
for filename, functions in required_files.items():
    if not os.path.exists(filename):
        missing_modules.append(f"{filename}")
        continue
    
    # 尝试导入并检查函数是否存在
    try:
        # 将路径转换为模块名：cs336_basics/check_points.py -> cs336_basics.check_points
        module_name = filename.replace('.py', '').replace('/', '.').replace('\\', '.')
        module = __import__(module_name, fromlist=[''])
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
from cs336_basics.check_points import save_checkpoint, load_checkpoint
from cs336_basics.data_utils import get_batch
from cs336_basics.model import TransformerLM, TransformerBlock, MultiHeadAttention, scaled_dot_product_attention
from cs336_basics.optimizer import AdamW, cosine_schedule_with_warmup, gradient_clipping
from cs336_basics.tokenizer import train_bpe, Tokenizer, get_tokenizer

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
    # 学习率扫描参数（用于问题a和b）
    parser.add_argument("--lr-sweep", action="store_true",
                        help="启用学习率扫描模式，测试多个学习率")
    parser.add_argument("--lr-values", type=str, default="1e-4,3e-4,1e-3,3e-3,1e-2",
                        help="逗号分隔的学习率列表，如 '1e-4,3e-4,1e-3'")
    parser.add_argument("--divergence-threshold", type=float, default=10.0,
                        help="损失超过此值视为发散（用于问题b）")
    parser.add_argument("--patience", type=int, default=3,
                        help="验证损失不改善的容忍轮数，用于早停")

    
    
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
    parser.add_argument("--checkpoint-dir", type=str, default="./datasets/checkpoints",
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
        # Batch size扫描参数（用于batch_size实验）
    parser.add_argument("--batch-size-sweep", action="store_true",
                        help="启用batch size扫描模式")
    parser.add_argument("--batch-size-values", type=str, 
                        default="1,4,8,16,32,64,128",
                        help="逗号分隔的batch size列表，如 '1,8,32,64'")
    parser.add_argument("--scale-lr-with-batchsize", action="store_true",
                        help="是否随batch size线性缩放学习率（推荐启用）")
    parser.add_argument("--scale-tokens-with-batchsize", action="store_true",
                    help="随batch size线性缩放total_tokens，保持迭代次数一致")


    return parser.parse_args()

def train_with_batch_size(args, batch_size_value, lr_value, checkpoint_subdir, wandb_group=None):
    """
    使用指定batch size训练模型，返回最终结果
    """
    from pathlib import Path
    import json
    # 在 train_with_batch_size 函数开头，计算num_iterations之前添加：
# 以batch_size=32为基准，按比例调整total_tokens
    BASE_BATCH_SIZE = 32
    if args.scale_tokens_with_batchsize:
        scaled_total_tokens = args.total_tokens * (batch_size_value / BASE_BATCH_SIZE)
    else:
        scaled_total_tokens = args.total_tokens  # 或设置一个上限，如min(args.total_tokens, 50_000_000)

    num_iterations = int(calculate_num_iterations(scaled_total_tokens, batch_size_value, args.context_length))


    # 创建专属目录
    os.makedirs(checkpoint_subdir, exist_ok=True)
    
    # 重新初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)
    
    # 优化器使用传入的学习率（如果启用线性缩放，lr_value已经缩放过了）
    optimizer = AdamW(
        model.parameters(),
        lr=lr_value,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    
    # 初始化W&B
    if args.use_wandb:
        run_name = f"{args.wandb_run_name}_bs{batch_size_value}" if args.wandb_run_name else f"bs{batch_size_value}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=wandb_group or "bs_sweep",
            config={**vars(args), "batch_size_actual": batch_size_value, "lr_actual": lr_value},
            reinit=True
        )
        wandb.watch(model, log="all", log_freq=args.log_interval)
    
    # 加载数据
    train_data = load_dataset_memory_efficient(args.train_data, dtype=np.uint16)
    val_data = load_dataset_memory_efficient(args.val_data, dtype=np.uint16)
    
    # 根据新的batch size重新计算迭代次数
    
    model.train()
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "iterations": []}
    
    print(f"\n{'='*60}")
    print(f"开始训练: batch_size = {batch_size_value}, lr = {lr_value:.2e}")
    print(f"总迭代次数: {num_iterations}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        x, y = get_batch(train_data, batch_size_value, args.context_length, args.device)
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        
        optimizer.zero_grad()
        loss.backward()
        
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        
        # 学习率调度
        lr = cosine_schedule_with_warmup(
            t=iteration, alpha_max=lr_value, alpha_min=args.min_lr,
            T_w=args.warmup_iters, T_c=num_iterations
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        optimizer.step()
        
        # 记录日志
        if iteration % args.log_interval == 0:
            history["train_loss"].append(loss.item())
            history["iterations"].append(iteration)
            if args.use_wandb:
                wandb.log({"train_loss": loss.item(), "learning_rate": lr}, step=iteration)
            
            elapsed = time.time() - start_time
            print(f"Iter {iteration:6d}/{num_iterations} | Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | Time: {elapsed:.1f}s")
        
        # 验证
        if iteration % args.val_interval == 0:
            val_loss = evaluate_model(model, val_data, batch_size_value, 
                                    args.context_length, args.device)
            history["val_loss"].append((iteration, val_loss))
            
            if args.use_wandb:
                wandb.log({"val_loss": val_loss, "val_ppl": torch.exp(torch.tensor(val_loss)).item()}, 
                         step=iteration)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_subdir, "best.pt")
                save_checkpoint(model, optimizer, iteration, best_path)
            
            print(f"Iter {iteration:6d} | Val Loss: {val_loss:.4f}")
    
    # 保存结果
    history_path = os.path.join(checkpoint_subdir, "history.json")
    with open(history_path, 'w') as f:
        json.dump({
            "batch_size": batch_size_value,
            "lr": lr_value,
            "best_val_loss": best_val_loss,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "history": history,
            "total_time": time.time() - start_time
        }, f, indent=2)
    
    if args.use_wandb:
        wandb.finish()
    
    return {
        "batch_size": batch_size_value,
        "lr": lr_value,
        "best_val_loss": best_val_loss,
        "checkpoint_dir": checkpoint_subdir
    }
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

def train_with_lr(args, lr_value, checkpoint_subdir, wandb_group=None):
    """
    使用指定学习率训练模型，返回最终验证损失和是否发散
    """
    from pathlib import Path
    import json
    
    # 创建该学习率的专属目录
    os.makedirs(checkpoint_subdir, exist_ok=True)
    
    # 重新初始化模型（每个学习率都需要全新的模型）
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=lr_value,  # 使用传入的学习率
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    
    # 初始化W&B（使用group来组织同一批实验）
    if args.use_wandb:
        run_name = f"{args.wandb_run_name}_lr{lr_value}" if args.wandb_run_name else f"lr{lr_value}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=wandb_group or "lr_sweep",
            config={**vars(args), "lr_actual": lr_value},
            reinit=True
        )
        wandb.watch(model, log="all", log_freq=args.log_interval)
    
    # 加载数据
    train_data = load_dataset_memory_efficient(args.train_data, dtype=np.uint16)
    val_data = load_dataset_memory_efficient(args.val_data, dtype=np.uint16)
    
    num_iterations = calculate_num_iterations(args.total_tokens, args.batch_size, args.context_length)
    
    # 训练状态
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    is_diverged = False
    history = {"train_loss": [], "val_loss": [], "iterations": []}
    
    print(f"\n{'='*60}")
    print(f"开始训练: 学习率 = {lr_value}")
    print(f"{'='*60}")
    
    for iteration in range(num_iterations):
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        
        # ===== 发散检测（关键用于问题b） =====
        if torch.isnan(loss) or loss.item() > args.divergence_threshold:
            print(f"⚠️  检测到发散！Loss={loss.item():.4f} (阈值: {args.divergence_threshold})")
            is_diverged = True
            break
        
        optimizer.zero_grad()
        loss.backward()
        
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
        
        # 学习率调度（注意：这里使用传入的lr_value作为alpha_max）
        lr = cosine_schedule_with_warmup(
            t=iteration, alpha_max=lr_value, alpha_min=args.min_lr,
            T_w=args.warmup_iters, T_c=num_iterations
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        optimizer.step()
        
        # 记录历史
        if iteration % args.log_interval == 0:
            history["train_loss"].append(loss.item())
            history["iterations"].append(iteration)
            if args.use_wandb:
                wandb.log({"train_loss": loss.item(), "learning_rate": lr}, step=iteration)
        
        # 验证
        if iteration % args.val_interval == 0:
            val_loss = evaluate_model(model, val_data, args.batch_size, 
                                    args.context_length, args.device)
            history["val_loss"].append((iteration, val_loss))
            
            print(f"Iter {iteration:6d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            
            if args.use_wandb:
                wandb.log({"val_loss": val_loss, "val_ppl": torch.exp(torch.tensor(val_loss)).item()}, 
                         step=iteration)
            
            # 早停检查（可选，用于提高效率）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                best_path = os.path.join(checkpoint_subdir, "best.pt")
                save_checkpoint(model, optimizer, iteration, best_path)
            else:
                patience_counter +=1
                if patience_counter >= args.patience:
                    print(f"早停触发（{args.patience}次验证无改善）")
                    break
    
    # 保存训练历史到JSON（用于后续绘制学习曲线）
    history_path = os.path.join(checkpoint_subdir, "history.json")
    with open(history_path, 'w') as f:
        json.dump({
            "lr": lr_value,
            "diverged": is_diverged,
            "best_val_loss": best_val_loss if not is_diverged else None,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "history": history
        }, f, indent=2)
    
    if args.use_wandb:
        wandb.finish()
    
    return {
        "lr": lr_value,
        "diverged": is_diverged,
        "best_val_loss": best_val_loss if not is_diverged else float('inf'),
        "checkpoint_dir": checkpoint_subdir
    }


def main():
    args = parse_args()
    
    if args.lr_sweep:
        # ===== 学习率扫描模式（用于完成作业） =====
        lr_list = [float(x.strip()) for x in args.lr_values.split(',')]
        results = []
        
        print(f"启动学习率扫描，共测试 {len(lr_list)} 个学习率: {lr_list}")
        
        for i, lr in enumerate(lr_list):
            # 为每个学习率创建子目录
            subdir = os.path.join(args.checkpoint_dir, f"lr_{lr:.0e}")
            
            # 运行训练
            result = train_with_lr(args, lr, subdir, wandb_group="lr_sweep_assignment")
            results.append(result)
            
            print(f"\n[{i+1}/{len(lr_list)}] 学习率 {lr} 完成:")
            print(f"   发散: {result['diverged']}")
            print(f"   最佳验证损失: {result['best_val_loss']:.4f}" if not result['diverged'] else "   N/A")
        
        # 生成汇总报告
        print("\n" + "="*60)
        print("学习率扫描结果汇总")
        print("="*60)
        print(f"{'学习率':<12} {'发散':<8} {'最佳Val Loss':<15} {'目标(≤1.45)':<10}")
        print("-"*60)
        
        for r in results:
            status = "✓" if (not r['diverged'] and r['best_val_loss'] <= 1.45) else "✗"
            val_str = f"{r['best_val_loss']:.4f}" if not r['diverged'] else "DIVERGED"
            print(f"{r['lr']:<12.0e} {str(r['diverged']):<8} {val_str:<15} {status:<10}")
        
        # 保存汇总结果
        import json
        summary_path = os.path.join(args.checkpoint_dir, "lr_sweep_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n详细结果已保存至: {summary_path}")
    elif args.batch_size_sweep:
        # ===== Batch Size扫描模式 =====
        bs_list = [int(x.strip()) for x in args.batch_size_values.split(',')]
        results = []
        
        # 基础学习率（可以根据batch size调整）
        base_lr = args.lr
        
        print(f"启动Batch Size扫描，共测试 {len(bs_list)} 个batch size: {bs_list}")
        
        for i, bs in enumerate(bs_list):
            # 为每个batch size计算对应的学习率（线性缩放）
            if args.scale_lr_with_batchsize:
                # 线性缩放规则：lr ∝ batch_size
                # 以bs=32为基准进行缩放
                lr_value = base_lr * (bs / 32.0)
            else:
                lr_value = base_lr  # 使用固定学习率
            
            # 创建子目录
            subdir = os.path.join(args.checkpoint_dir, f"bs_{bs}_lr_{lr_value:.0e}")
            
            # 运行训练
            result = train_with_batch_size(args, bs, lr_value, subdir, 
                                         wandb_group="bs_sweep_assignment")
            results.append(result)
            
            print(f"\n[{i+1}/{len(bs_list)}] Batch Size {bs} 完成:")
            print(f"   学习率: {lr_value:.2e}")
            print(f"   最佳验证损失: {result['best_val_loss']:.4f}")
            print(f"   等效batch数: {args.total_tokens / (bs * args.context_length):,.0f} iters")
        
        # 生成汇总报告
        print("\n" + "="*60)
        print("Batch Size扫描结果汇总")
        print("="*60)
        print(f"{'Batch Size':<12} {'LR':<12} {'最佳Val Loss':<15} {'总迭代次数':<12} {'状态':<10}")
        print("-"*60)
        
        for r in results:
            num_iter = args.total_tokens // (r['batch_size'] * args.context_length)
            status = "✓" if r['best_val_loss'] <= 1.45 else "✗"
            print(f"{r['batch_size']:<12} {r['lr']:<12.0e} {r['best_val_loss']:<15.4f} {num_iter:<12,} {status:<10}")
        
        # 保存汇总结果
        summary_path = os.path.join(args.checkpoint_dir, "bs_sweep_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n详细结果已保存至: {summary_path}")   


    else:
        # ===== 原始单学习率训练模式 =====
        # 这里保持你原有的main函数逻辑不变
        args_single = vars(args)
        args_single['lr_values'] = str(args.lr)  # 只测试默认学习率
        result = train_with_lr(args, args.lr, args.checkpoint_dir)
        print(f"训练完成，最佳验证损失: {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
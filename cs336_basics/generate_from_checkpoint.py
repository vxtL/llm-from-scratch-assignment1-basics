# generate_from_checkpoint.py
import torch
import json
import argparse
import sys
from pathlib import Path
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import get_tokenizer
from cs336_basics.generate import decode

def infer_model_config(checkpoint_path):
    """
    从检查点文件智能推断模型配置（适配你的训练参数）
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # 从token_embeddings推断
    vocab_size = state_dict['token_embeddings.weight'].shape[0]
    d_model = state_dict['token_embeddings.weight'].shape[1]
    
    # 推断层数（与你的训练脚本一致）
    num_layers = sum(1 for key in state_dict.keys() 
                     if key.startswith('layers.') and '.ln1.weight' in key)
    
    # 推断num_heads：从你的训练默认值16调整
    # 如果d_model=512, num_heads=16 => head_dim=32
    # 如果d_model=768, num_heads=12 => head_dim=64
    if d_model == 768:
        num_heads = 12
    elif d_model == 512:
        num_heads = 16
    else:
        num_heads = max(8, d_model // 64)  # 智能推断
    
    # 推断d_ff：从你的训练默认值1344
    d_ff = None
    for key in state_dict:
        if 'layers.0.ffn.w1.weight' in key:
            d_ff = state_dict[key].shape[0]
            break
    
    if d_ff is None:
        d_ff = d_model * 4  # 默认值
    
    # 上下文长度从positional encoding或第一层推断
    context_length = 256  # 你的训练默认值
    
    config = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": 10000.0,
    }
    
    print("=" * 60)
    print("从检查点自动推断的模型配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    
    return config

def load_checkpoint_with_config(checkpoint_path, device='auto'):
    """
    加载检查点并自动推断配置
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"📦 正在加载检查点: {checkpoint_path}")
    print(f"💻 使用设备: {device}")
    
    # 推断配置
    config = infer_model_config(checkpoint_path)
    
    # 创建并加载模型
    model = TransformerLM(**config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ 模型加载完成（迭代次数: {checkpoint['iteration']}）")
    
    return model, config

# 在文件中找到这个函数，替换 else 分支的内容
def load_tokenizer_from_training(vocab_path, merges_path):
    """
    兼容训练脚本的分词器加载（处理特殊token格式）
    """
    print(f"📦 正在加载分词器...")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    with open(merges_path, 'r', encoding='utf-8') as f:
        merges_data = json.load(f)
    
    # 转换格式（处理特殊token如 <|endoftext|>）
    vocab_dict = {}
    for v, k in vocab_data.items():
        token_id = int(k)
        if isinstance(v, str):
            if v.startswith('<') and v.endswith('>'):
                # 特殊token保持原样
                token_bytes = v.encode('utf-8')
            else:
                # 处理unicode转义，失败时回退到原始编码
                try:
                    token_bytes = v.encode('utf-8').decode('unicode_escape').encode('utf-8')
                except UnicodeDecodeError:
                    token_bytes = v.encode('utf-8')
        else:
            # 如果v是整数（字节值），直接转为字节
            token_bytes = bytes([v]) if isinstance(v, int) else bytes(v)
        vocab_dict[token_id] = token_bytes
    
    # 转换merges（保持原有逻辑）
    merges_list = []
    for pair in merges_data:
        if isinstance(pair, list) and len(pair) == 2:
            first = pair[0].encode('utf-8') if isinstance(pair[0], str) else bytes([pair[0]])
            second = pair[1].encode('utf-8') if isinstance(pair[1], str) else bytes([pair[1]])
            merges_list.append((first, second))
    
    tokenizer = get_tokenizer(vocab_dict, merges_list, special_tokens=["<|endoftext|>"])
    print(f"✅ 分词器加载完成（词汇表大小: {len(vocab_dict)}）")
    
    return tokenizer

def main():
    parser = argparse.ArgumentParser(description="从best.pt生成文本")
    
    # 输入文件（根据你的训练脚本结构）
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="检查点路径（如 checkpoints/lr_1e-4/best.pt）")
    parser.add_argument("--vocab", type=str, default="vocab.json",
                        help="vocab.json路径")
    parser.add_argument("--merges", type=str, default="merges.json",
                        help="merges.json路径")
    
    # 生成参数
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is",
                        help="提示词")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    
    # 输出
    parser.add_argument("--output", type=str, default=None,
                        help="保存结果的文件路径")
    
    args = parser.parse_args()
    
    # 验证文件
    for path in [args.checkpoint, args.vocab, args.merges]:
        if not Path(path).exists():
            print(f"❌ 错误：文件不存在 - {path}")
            sys.exit(1)
    
    # 加载组件
    model, config = load_checkpoint_with_config(args.checkpoint, args.device)
    tokenizer = load_tokenizer_from_training(args.vocab, args.merges)
    
    # 生成
    print(f"\n📝 生成参数: temp={args.temperature}, top_p={args.top_p}")
    print(f"💬 提示词: '{args.prompt}'")
    print("=" * 80)
    
    with torch.no_grad():
        full_text, generated_ids = decode(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            end_token="<|endoftext|>",
            device=next(model.parameters()).device
        )
    
    print(full_text)
    print("=" * 80)
    
    # 统计
    print(f"\n📊 生成统计:")
    print(f"- Token数量: {len(generated_ids)}")
    print(f"- 生成字符数: {len(full_text) - len(args.prompt)}")
    print(f"- 遇到结束符: {'<|endoftext|>' in full_text}")
    
    # 保存
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"\n💾 结果已保存: {args.output}")
    
    return full_text

def batch_generate_comparison():
    """
    批量生成对比（用于找出最佳参数）
    """
    # 参数组合实验
    configs = [
        {"temp": 0.5, "top_p": 0.95, "desc": "保守-高质量"},
        {"temp": 0.8, "top_p": 0.9, "desc": "平衡-推荐"},
        {"temp": 1.0, "top_p": 0.85, "desc": "创造-多样性"},
    ]
    
    results = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {cfg['desc']} (temp={cfg['temp']}, top_p={cfg['top_p']})")
        print('='*60)
        
        # 这里调用main()或生成函数
        # 为简洁起见，省略具体实现

if __name__ == "__main__":
    main()
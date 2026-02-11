#!/usr/bin/env python3
"""
将 JSON 文件中的 "Ġ" 替换为 " "（空格）
常用于处理 GPT-2/BPE tokenizer 的 vocab.json
"""

import json
import sys
import argparse
from pathlib import Path


def replace_g_in_json(input_path, output_path=None, inplace=False):
    """
    递归替换 JSON 对象中的所有字符串中的 "Ġ" 为 " "
    
    Args:
        input_path: 输入 JSON 文件路径
        output_path: 输出文件路径（如果 inplace=True 则忽略）
        inplace: 是否直接覆盖原文件
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    # 读取 JSON
    print(f"读取: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 递归替换
    def replace_recursive(obj):
        if isinstance(obj, str):
            # 核心替换：Ġ → 空格
            return obj.replace('Ġ', ' ')
        elif isinstance(obj, dict):
            return {replace_recursive(k): replace_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_recursive(item) for item in obj]
        else:
            return obj
    
    print("替换 'Ġ' → ' ' ...")
    new_data = replace_recursive(data)
    
    # 统计替换次数（粗略估计）
    def count_g(obj):
        if isinstance(obj, str):
            return obj.count('Ġ')
        elif isinstance(obj, dict):
            return sum(count_g(k) + count_g(v) for k, v in obj.items())
        elif isinstance(obj, list):
            return sum(count_g(item) for item in obj)
        return 0
    
    g_count = count_g(data)
    print(f"  共找到 {g_count} 处 'Ġ' 标记")
    
    # 确定输出路径
    if inplace:
        output_path = input_path
    elif output_path is None:
        # 默认添加 .new 后缀
        output_path = input_path.with_suffix('.new.json')
    else:
        output_path = Path(output_path)
    
    # 写入
    print(f"写入: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print("完成！")


def main():
    parser = argparse.ArgumentParser(
        description='将 JSON 文件中的 "Ġ" (GPT BPE 空格标记) 替换为普通空格'
    )
    parser.add_argument('input', help='输入 JSON 文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（默认: input.new.json）')
    parser.add_argument('-i', '--inplace', action='store_true', 
                        help='直接覆盖原文件（谨慎使用）')
    
    args = parser.parse_args()
    
    try:
        replace_g_in_json(args.input, args.output, args.inplace)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
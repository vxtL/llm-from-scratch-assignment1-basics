import os
import typing
import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
):
    """
    保存模型、优化器状态和迭代次数到检查点文件
    
    参数:
        model: PyTorch 模型
        optimizer: PyTorch 优化器
        iteration: 当前迭代次数
        out: 输出路径或文件对象
    """
    # 构建要保存的状态字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # 保存到文件
    torch.save(checkpoint, out)


def load_checkpoint(
    src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    从检查点加载模型和优化器状态
    
    参数:
        src: 检查点文件路径或文件对象
        model: PyTorch 模型（将被原地修改）
        optimizer: PyTorch 优化器（将被原地修改）
    
    返回:
        保存的迭代次数
    """
    # 加载检查点
    checkpoint = torch.load(src)
    
    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 返回迭代次数
    return checkpoint['iteration']



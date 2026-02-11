from collections.abc import Callable,Iterable

from collections import defaultdict
from typing import Iterable, Optional,Union

from torch.optim import Optimizer
import torch
import math

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失。
    
    参数:
        logits: 预测的logits，形状为 (batch_size, vocab_size)
        targets: 目标类别索引，形状为 (batch_size,)
    
    返回:
        平均交叉熵损失（标量张量）
    """
    # 数值稳定性处理：减去每行的最大值
    # 在vocab_size维度上计算最大值并保持维度以便广播
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    stable_logits = logits - max_logits
    
    # 计算log softmax
    # 使用log-sum-exp技巧避免直接计算exp导致溢出
    exp_logits = torch.exp(stable_logits)
    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    log_softmax = stable_logits - log_sum_exp
    
    # 提取正确类别的log概率
    batch_size = logits.shape[0]
    # 使用arange和targets索引选择正确的log概率
    correct_log_probs = log_softmax[torch.arange(batch_size), targets]
    
    # 计算负的平均损失
    loss = -torch.mean(correct_log_probs)
    
    return loss

class SGD(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3):
        if lr<0:
            raise ValueError(f"Invalid learning rate:{lr}")
        defaults = {"lr":lr}
        super().__init__(params,defaults)
    
    def step(self,closure:Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
            
                state = self.state[p]
                t = state.get("t",0)
                grad = p.grad.data
                p.data -= lr/math.sqrt(t+1)*grad
                state["t"] = t+1
        return loss
    

class AdamW(Optimizer):
    """
    AdamW优化器实现，遵循Loshchilov和Hutter 2019的算法2。
    
    参数:
        params: 待优化的参数迭代器
        lr: 学习率 (默认: 1e-3)
        betas: (β1, β2) 元组，用于动量估计 (默认: (0.9, 0.999))
        eps: 数值稳定性项 (默认: 1e-8)
        weight_decay: 权重衰减系数λ (默认: 0.01)
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        # 验证超参数
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        # 设置默认参数字典
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        
        # 调用父类构造函数
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        执行一次优化步骤。
        
        参数:
            closure: 可选的闭包，用于重新计算损失
            
        返回:
            如果提供了closure，则返回closure的结果
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        # 遍历每个参数组
        for group in self.param_groups:
            # 从参数组中提取超参数
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            # 遍历该组中的每个参数
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 获取参数的梯度
                grad = p.grad.data
                
                # 获取或初始化参数状态
                state = self.state[p]
                
                # 状态初始化（如果这是第一次看到这个参数）
                if len(state) == 0:
                    # 初始化时间步
                    state['step'] = 0
                    # 初始化第一阶矩估计（动量）
                    state['m'] = torch.zeros_like(p.data)
                    # 初始化第二阶矩估计（未中心化的方差）
                    state['v'] = torch.zeros_like(p.data)
                
                # 获取当前状态
                m, v = state['m'], state['v']
                state['step'] += 1
                step = state['step']
                
                # 偏差校正系数
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # 更新第一阶矩估计（动量）
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 更新第二阶矩估计
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 计算当前步的学习率（包含偏差校正）
                denom = math.sqrt(bias_correction2) / bias_correction1
                step_size = lr * denom
                
                # 计算更新量
                # θ ← θ - α_t * m / (√v + ε)
                # 等价于: θ ← θ - step_size * m / (√v + ε * √bias_correction2)
                # 为了数值稳定性，我们在分母中使用eps
                v_sqrt = torch.sqrt(v)
                update = step_size * m / (v_sqrt + eps * math.sqrt(bias_correction2))
                
                # 更新参数
                # 注意：PyTorch的Optimizer要求原地修改p.data
                p.data.add_(update, alpha=-1)
                
                # 应用权重衰减
                # θ ← θ - α * λ * θ
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        
        return loss


def cosine_schedule_with_warmup(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int
) -> float:
    """
    实现余弦退火学习率调度，包含warmup阶段
    
    参数:
        t: 当前迭代次数（从0开始）
        alpha_max: 最大学习率
        alpha_min: 最小学习率
        T_w: warmup迭代次数
        T_c: 余弦退火总迭代次数
    
    返回:
        当前迭代对应的学习率
    """
    # Warm-up阶段：线性增长
    if t < T_w:
        return (t / T_w) * alpha_max
    
    # 余弦退火阶段：在T_w到T_c之间平滑下降
    elif t <= T_c:
        # 计算在余弦周期的进度（0到1之间）
        progress = (t - T_w) / (T_c - T_w)
        # 余弦退火因子
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return alpha_min + cosine_factor * (alpha_max - alpha_min)
    
    # Post-annealing阶段：保持最小学习率
    else:
        return alpha_min



def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float):
    """
    实现梯度裁剪（Gradient Clipping）
    
    参数:
        params: 模型参数的可迭代对象（通常是 model.parameters()）
        max_norm: 梯度的最大L2范数阈值
    
    功能:
        计算所有参数梯度的总L2范数，如果超过max_norm，
        则将所有梯度按比例缩小，使其总范数等于max_norm
    """
    # 收集所有非None的梯度
    gradients = [p.grad for p in params if p.grad is not None]
    
    if len(gradients) == 0:
        return  # 没有梯度需要裁剪
    
    # 计算所有梯度的总L2范数
    # 先计算每个梯度的范数，再计算这些范数的范数
    total_norm = torch.norm(torch.stack([torch.norm(g) for g in gradients]), p=2)
    
    # 计算裁剪系数
    clip_coef = max_norm / (total_norm + 1e-6)  # 添加epsilon防止除零
    
    # 如果总范数超过阈值，则裁剪
    if clip_coef < 1.0:
        # 原地修改梯度
        for g in gradients:
            g.mul_(clip_coef)  # 等价于 g *= clip_coef
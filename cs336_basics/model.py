import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional
from jaxtyping import Float, Int, Bool

# --- 辅助函数：手动实现 activation 和 softmax ---
def silu(x):
    return x * torch.sigmoid(x)

def softmax(x, dim=-1):
    # 减去最大值以保证数值稳定性
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    e_x = torch.exp(x - x_max)
    return e_x / torch.sum(e_x, dim=dim, keepdim=True)

class LayerNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dims))
        self.beta = nn.Parameter(torch.zeros(dims))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        # 使用无偏估计为 False 的方差，符合标准 LayerNorm 行为
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


# --- §3.1 & §3.2: 基础层 ---
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        # 1. 存储为 W (out_features, in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        # 2. 必须进行初始化 
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3. 使用行向量约定 y = xW^T 
        return x @ self.weight.t()

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        # 建议支持 device 和 dtype 参数 [cite: 206, 207, 208]
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        
        # 必须使用截断正态分布初始化 [cite: 198, 199, 209]
        # 均值=0, 标准差=1, 截断范围 [-3, 3]
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx 形状通常为 [batch_size, seq_len]
        # 输出形状为 [batch_size, seq_len, d_model]
        return self.weight[idx]

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # 建议通过 factory_kwargs 传递设备和类型
        self.weight = nn.Parameter(torch.ones(dims, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        # 公式: x * 1/sqrt(mean(x^2) + eps) * weight
        norm_factor = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm_factor * self.weight

# --- §3.4: RoPE ---
def apply_rope(x: torch.Tensor, theta: float = 10000.0, token_positions: torch.Tensor = None) -> torch.Tensor:
    device = x.device
    *batch_dims, seq_len, d = x.shape
    
    # 1. 计算频率
    dim_indices = torch.arange(0, d, 2, device=device).float()
    freqs = 1.0 / (theta ** (dim_indices / d))
    
    # 2. 获取位置索引
    if token_positions is None:
        t = torch.arange(seq_len, device=device).float()
    else:
        t = token_positions.float() # positions shape: [batch, seq_len] 或 [seq_len]

    # 3. 计算角度 [..., seq_len, d/2]
    # 注意处理广播：t 可能有 batch 维度
    angles = t.unsqueeze(-1) * freqs 
    
    cos = torch.cos(angles) # [..., seq_len, d/2]
    sin = torch.sin(angles)
    
    # 4. 旋转逻辑（请确认测试要求的是交错还是分半）
    # 以下为交错实现（Interleaved）的修正版
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    # 确保 cos/sin 形状与 x_even 匹配
    # 如果 angles 是 [batch, seq_len, d/2]，需要适配 head 维度
    if len(angles.shape) != len(x_even.shape):
        # 假设 x_even 是 [batch, num_heads, seq_len, head_dim/2]
        # angles 需要从 [batch, seq_len, d/2] 插一个维度变成 [batch, 1, seq_len, d/2]
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    
    out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    return out

# --- §3.3: SwiGLU ----
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        gate = self.w1(x)       # d_ff
        up = self.w3(x)         # d_ff
        activated = silu(gate) * up
        return self.w2(activated) # d_model

# --- §3.5: 初始化逻辑 ---
def init_weights(module, n_layers, d_model):
    # 处理 Linear 层 [cite: 98, 109]
    if isinstance(module, Linear):
        d_in = module.weight.size(1)
        d_out = module.weight.size(0)
        std = math.sqrt(2.0 / (d_in + d_out))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
        
    # 处理 Embedding 层 [cite: 99, 121]
    elif isinstance(module, Embedding):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        
    # 处理 RMSNorm 层 (如果有) 
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight) # 或者 self.gamma，取决于你的实现名

# --- 修改后的 TransformerBlock ---
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope_theta: float, dropout: float = 0.0, device=None, dtype=None):
        super().__init__()
        # 1. 使用 RMSNorm (PDF §3.6 明确要求)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, rope_theta, device=device, dtype=dtype)
        
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        
        # 建议添加 Dropout (根据标准 Transformer 和测试要求)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        mask: 预先生成的因果掩码 [1, 1, seq_len, seq_len]
        """
        # 如果外部没有传入 mask，可以在此处生成，但建议由上层 TransformerLM 传入
        if mask is None:
            l = x.size(1)
            mask = torch.tril(torch.ones(l, l, device=x.device, dtype=torch.bool)).view(1, 1, l, l)

        # Pre-norm Attention (PDF Eq. 15: y = x + MHA(RMSNorm(x)))
        # 
        normed_x = self.ln1(x)
        attn_out = self.attn(normed_x, mask=mask)
        x = x + self.dropout(attn_out) # 加上残差和 Dropout
        
        # Pre-norm FFN
        normed_x = self.ln2(x)
        ff_out = self.ffn(normed_x)
        x = x + self.dropout(ff_out) # 加上残差和 Dropout
        
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, rope_theta: float = 10000.0,
                 device=None, dtype=None): # 添加 device 和 dtype 支持 [cite: 98]
        super().__init__()
        # 透传 device 和 dtype
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope_theta, device=device, dtype=dtype) 
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
        # 应用初始化
        self.apply_custom_init(num_layers, d_model)

    def apply_custom_init(self, n_layers, d_model):
        for name, m in self.named_modules():
            if isinstance(m, Linear):
                # 按照 PDF §3.4.1 的公式: std = sqrt(2 / (d_in + d_out)) 
                d_in = m.weight.size(1)
                d_out = m.weight.size(0)
                std = math.sqrt(2.0 / (d_in + d_out))
                
                # 如果遵循特定作业阶段对残留路径的特殊缩放 (§3.5)
                if 'output_proj' in name or 'ffn.w2' in name:
                    std = std / math.sqrt(2 * n_layers)
            
                # 截断范围为 [-3sigma, 3sigma] 
                nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-3*std, b=3*std)
            
            elif isinstance(m, Embedding):
                # Embedding 初始化: N(0, 1), 截断范围 [-3, 3] 
                nn.init.trunc_normal_(m.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
            
            elif isinstance(m, RMSNorm):
                # RMSNorm 初始化为 1 
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: [batch, seq_len]
        x = self.token_embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))


    
    


# --- 新增：Multi-Head Attention 子模块 ---
# 在 model.py 中重写 MultiHeadAttention 的 forward
# model.py 核心修改部分

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        rope_theta: Optional[float] = None, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_theta = rope_theta 
        
        # 投影层：按照 PDF 要求不使用 bias
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # --- 关键补充：PDF §3.3 要求的初始化 ---
        self.reset_parameters()

    def reset_parameters(self):
        # 使用截断正态分布初始化，标准差通常设为 0.02
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.output_proj]:
            if hasattr(layer, 'W'):
                torch.nn.init.trunc_normal_(layer.W, std=0.02)

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None, 
        token_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, l, d = x.shape
        
        # 1. 投影并重塑为 [batch, num_heads, seq_len, head_dim]
        # 注意：view 顺序是 (b, l, h, d_h)，然后 transpose(1, 2) 变为 (b, h, l, d_h)
        q = self.q_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. 应用 RoPE (仅针对 Q 和 K)
        if self.rope_theta is not None:
            q = apply_rope(q, self.rope_theta, token_positions=token_positions)
            k = apply_rope(k, self.rope_theta, token_positions=token_positions)
        
        # 3. 默认下三角因果掩码 (Causal Mask)
        if mask is None:
            mask = torch.tril(torch.ones(l, l, device=x.device, dtype=torch.bool)).view(1, 1, l, l)
        
        # 4. 缩放点积注意力
        # 请确保你的 scaled_dot_product_attention 内部使用了 1/sqrt(head_dim) 的缩放
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # 5. 恢复形状 [batch, seq_len, d_model] 并进行输出投影
        out = out.transpose(1, 2).contiguous().view(b, l, d)
        return self.output_proj(out)
    

# model.py

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0, # PDF §3.3 提到的参数
    training: bool = False,
) -> torch.Tensor:
    # 1. 计算缩放点积分数
    d_k = q.size(-1)
    # scores 形状: [batch, num_heads, seq_len, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. 应用 Mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    
    # 3. 稳定版 Softmax
    attn_weights = softmax(scores, dim=-1)
    
    # 4. 应用 Dropout (PDF §3.3 要求)
    if dropout_p > 0.0:

        attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p, training=training)
    
    # 5. 加权求和
    return torch.matmul(attn_weights, v)



#class MultiHeadSelfAttentionWithRoPE(nn.Module):
#    """纯多头自注意力（含RoPE），不含残差和FFN"""
#    def __init__(self, d_model: int, num_heads: int, rope_theta: float):
#        super().__init__()
#        self.num_heads = num_heads
#        self.head_dim = d_model // num_heads
#        self.rope_theta = rope_theta
#        
#        self.q_proj = Linear(d_model, d_model)
#        self.k_proj = Linear(d_model, d_model)
#        self.v_proj = Linear(d_model, d_model)
#        self.output_proj = Linear(d_model, d_model)
#        
#    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
#        # 1. 投影
#        q = self.q_proj(x)
#        k = self.k_proj(x)
#        v = self.v_proj(x)
#        
#        # 2. 拆分多头并转置
#        batch_size, seq_len, _ = x.shape
#        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#        
#        # 3. 应用RoPE（复用model.py的apply_rope）
#        q = apply_rope(q, self.rope_theta)
#        k = apply_rope(k, self.rope_theta)
#        
#        # 4. 因果掩码
#        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
#        mask = mask.view(1, 1, seq_len, seq_len)
#        
#        # 5. 缩放点积注意力（复用model.py的softmax）
#        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#        scores = scores.masked_fill(mask == 0, float('-inf'))
#        attn = softmax(scores, dim=-1) @ v
#        
#        # 6. 合并多头并输出投影
#        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
#        return self.output_proj(attn)
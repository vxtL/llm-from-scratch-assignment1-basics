from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
from cs336_basics.tokenizer import train_bpe
from cs336_basics.model import (
    Linear, Embedding, RMSNorm, SwiGLU, 
    TransformerBlock, TransformerLM, apply_rope, softmax,silu
)
from cs336_basics.model import MultiHeadAttention

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange
import math

def run_linear(d_in: int, d_out: int, weights: torch.Tensor, in_features: torch.Tensor) -> torch.Tensor:
    layer = Linear(d_in, d_out)
    layer.weight = torch.nn.Parameter(weights)
    return layer(in_features)


def run_embedding(vocab_size: int, d_model: int, weights: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    emb = Embedding(vocab_size, d_model)
    emb.weight = torch.nn.Parameter(weights)
    return emb(token_ids)


def run_swiglu(d_model: int, d_ff: int, w1_weight: torch.Tensor, w2_weight: torch.Tensor, w3_weight: torch.Tensor, in_features: torch.Tensor) -> torch.Tensor:
    ffn = SwiGLU(d_model, d_ff)
    ffn.w1.weight = torch.nn.Parameter(w1_weight)
    ffn.w2.weight = torch.nn.Parameter(w2_weight)
    ffn.w3.weight = torch.nn.Parameter(w3_weight)
    return ffn(in_features)


# adapters.py

# adapters.py

def run_scaled_dot_product_attention(
    Q: Float[Tensor, "batch num_heads seq_len d_k"],
    K: Float[Tensor, "batch num_heads seq_len d_k"],
    V: Float[Tensor, "batch num_heads seq_len d_k"],
    mask: Bool[Tensor, "batch num_heads seq_len seq_len"] | None = None,
) -> Float[Tensor, "batch num_heads seq_len d_k"]:
    """
    胶水代码：将 Q, K, V (大写) 转发给 model.py 中的实现 (小写)
    """
    from cs336_basics.model import scaled_dot_product_attention
    # 转发时将大写变量传给函数对应的参数
    return scaled_dot_product_attention(q=Q, k=K, v=V, mask=mask)

# adapters.py


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor, # 修改这里：将 output_proj_weight 改为 o_proj_weight
    in_features: torch.Tensor,
) -> torch.Tensor:
    """
    纯胶水代码：实例化 MultiHeadAttention 并手动加载测试权重
    """


    # 1. 实例化模型 (这个测试不带 RoPE，所以传 None)
    model = MultiHeadAttention(d_model, num_heads, rope_theta=None)

    # 2. 加载投影层权重
    # 映射测试传入的名称到模型内部的层名
    model.q_proj.weight = torch.nn.Parameter(q_proj_weight)
    model.k_proj.weight = torch.nn.Parameter(k_proj_weight)
    model.v_proj.weight = torch.nn.Parameter(v_proj_weight)
    model.output_proj.weight = torch.nn.Parameter(o_proj_weight)

   
    # 4. 调用模型
    return model(in_features)



def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    # 纯胶水：加载状态字典 -> 转发
    model = MultiHeadAttention(d_model, num_heads, rope_theta=theta)
    model.load_state_dict({
        'q_proj.weight': q_proj_weight,
        'k_proj.weight': k_proj_weight,
        'v_proj.weight': v_proj_weight,
        'output_proj.weight': o_proj_weight,
    })
    
    return model(in_features) # 同样不再传 mask


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, torch.Tensor],
    in_features: torch.Tensor,
) -> torch.Tensor:
    # 实例化
    block = TransformerBlock(d_model, num_heads, d_ff, rope_theta=theta)
    
    # 加载权重
    block.load_state_dict(weights, strict=True)
    block.eval()
    
    with torch.no_grad():
        # TransformerBlock.forward 内部现在已经包含了正确的 mask 生成逻辑
        return block(in_features)


# adapters.py

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, torch.Tensor],
    in_indices: torch.Tensor,
) -> torch.Tensor:
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )

    model.load_state_dict(weights, strict=True)
    model.eval()

    with torch.no_grad():
        # 移除 token_positions=token_positions 参数
        return model(in_indices)


def run_rmsnorm(d_model: int, eps: float, weights: torch.Tensor, in_features: torch.Tensor) -> torch.Tensor:
    norm = RMSNorm(d_model, eps=eps)
    norm.weight = torch.nn.Parameter(weights)
    return norm(in_features)


def run_silu(in_features: torch.Tensor) -> torch.Tensor:
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


# adapters.py

def run_softmax(
    in_features: Float[Tensor, " ... d_in"],
    dim: int = -1,  # 添加 dim 参数以匹配测试调用
) -> Float[Tensor, " ... d_in"]:
    """
    胶水代码：将测试转发给 model.py 中的实现
    """
    from cs336_basics.model import softmax
    # 将 dim 转发给 model.py 中的 softmax 函数
    return softmax(in_features, dim=dim)

def run_rope(
    d_model: int, 
    theta: float, 
    max_seq_len: int, 
    in_query_or_key: torch.Tensor, 
    token_positions: torch.Tensor = None
) -> torch.Tensor:
    """
    适配器：现在完整转发 token_positions 给 model.py
    """
    from cs336_basics.model import apply_rope
    # 注意：在 RoPE 测试中，in_query_or_key 通常已经是 (batch, heads, seq, head_dim)
    return apply_rope(in_query_or_key, theta=theta, token_positions=token_positions)

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    from cs336_basics.tokenizer import Tokenizer
    return Tokenizer(vocab, merges, special_tokens)



def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    适配器：把 CS336 官方接口直接转发到我们自己的 train_bpe 实现。
    """
    # 2. 路径转成 str（PathLike 也支持）；其余参数原样传
    return train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )






def run_layer_norm(
    dims: int,
    eps: float,
    gamma: Float[Tensor, " dims"],
    beta: Float[Tensor, " dims"],
    in_features: Float[Tensor, " ... dims"],
) -> Float[Tensor, " ... dims"]:
    """
    胶水代码：实例化 model.py 中的 LayerNorm 并加载权重转发请求
    """
    from cs336_basics.model import LayerNorm
    
    # 1. 实例化模型
    ln = LayerNorm(dims, eps=eps)
    
    # 2. 加载测试传入的权重
    ln.gamma = torch.nn.Parameter(gamma)
    ln.beta = torch.nn.Parameter(beta)
    
    # 3. 运行前向传播
    return ln(in_features)
from __future__ import annotations

import math
from typing import Optional

from einops import rearrange
from jaxtyping import Float, Int
import torch
from torch import Tensor
from torch import nn
import einops


class Linear(nn.Module):
    """线性层，实现 y = W x。

    本模块的权重张量形状为 (out_features, in_features)。在前向传播中，
    通过对权重做转置后进行矩阵乘法，以支持输入张量的任意前导批量维度。
    """

    in_features: int
    out_features: int
    weight: nn.Parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 按形状 (out_features, in_features) 创建可学习权重参数
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # 使用截断正态分布初始化，标准差 std = sqrt(2 / (in_features + out_features))
        variance = 2.0 / float(in_features + out_features)
        std = math.sqrt(variance)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0, b=3.0)

    def forward(self, x: Tensor) -> Tensor:
        # 期望输入 x 的形状为 (..., in_features)
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"期望输入最后一维为 {self.in_features}, 实际为 {x.shape[-1]}"
            )
        # 计算 y = x @ W^T，输出形状为 (..., out_features)
        return einops.einsum(x, self.weight, "... in_features, out_features in_features ->... out_features")


class Embedding(nn.Module):

    num_embeddings: int
    embedding_dim: int
    weight: nn.Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # weight：(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (batch_size, seq_len)
        # weight: (num_embeddings, embedding_dim)
        # output: (batch_size, seq_len, embedding_dim)
        # 以token_ids为索引，从weight中取出对应的embedding，索引纬度是num_embeddings
        return self.weight[token_ids]


class RMSNorm(nn.Module):

    d_model: int
    eps: float
    g: nn.Parameter

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5, # ϵ为固定值
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, seq_len, d_model)
        # 将x转换为float32类型，返回的时候还原
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 计算RSMNorm
        square = x ** 2  # 逐元素平方
        mean_square = square.mean(dim=-1, keepdim=True)
        rsm = torch.sqrt(mean_square + self.eps)
        result = x / rsm * self.g

        return result.to(in_dtype)

class PositionWiseFFN(nn.Module):

    d_model: int
    d_ff: int
    W1: Linear
    W2: Linear
    W3: Linear

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = int(8/3*d_model)
            self.d_ff = ((self.d_ff + 63) // 64) * 64  # 向上取整为64的倍数
        else:
            self.d_ff = d_ff
        # Linear入参顺序是 （out_features, in_features），所以初始化跟公式是反过来的
        self.W1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.W2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        w1_x = self.W1(x)
        w3_x = self.W3(x)
        SiLU = w1_x * torch.sigmoid(w1_x)
        GLU = SiLU * w3_x
        return self.W2(GLU)

class RoPE(nn.Module):
    theta: float
    d_k: int
    max_seq_len: int

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        *,
        device: Optional[torch.device | str] = None
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        if self.d_k % 2 != 0:
            raise ValueError(f"d_k必须是偶数（文档要求拆成2D向量对），d_k = {self.d_k}")

        # 预计算“元素对的角度系数”（对应文档中的i/Θ^((2k-1)/d_k)）
        k = torch.arange(0, self.d_k//2, device=device, dtype=torch.float32)
        theta_coeff = theta ** (2*k / d_k)
        theta_coeff = 1 / theta_coeff
        # 生成所有可能的位置i
        positions = torch.arange(0,max_seq_len, device=device, dtype=torch.float32)

        # 计算每个位置i、每个元素对k的角度θ = i * 系数（文档中的θ_{i,k}）
        angles = einops.einsum(positions, theta_coeff, "seq_pos, k -> seq_pos k")

        # 预计算cos和sin，存成缓存（文档要求用register_buffer，不参与训练）
        cos_cache = torch.cos(angles)  # 形状：(max_seq_len, d_k//2)
        sin_cache = torch.sin(angles)  # 形状：(max_seq_len, d_k//2)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    # 输入：x -> (..., seq_len, d_k), token_positions -> (..., seq_len)
    # 输出：x -> (..., seq_len, d_k)
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 按照 1缓存切片→2向量旋转→3维度重组 的顺序，对x进行处理

        batch_dims = token_positions.shape[:-1]
        seq_len = token_positions.shape[-1]

        # 1 缓存切片
        # 1.1 将token_positions展平，得到 (flat_batch, seq_len)
        token_positions_flat = rearrange(token_positions, "... s -> (...) s")
        # 1.2 从缓存中取出cos和sin，得到 (flat_batch, seq_len, d_k//2)
        cos_cache = self.cos_cache[token_positions_flat]
        sin_cache = self.sin_cache[token_positions_flat]
        # 1.3 还原flat_batch,将cos和sin的形状从(flat_batch, seq_len, d_k//2) → (..., seq_len, d_k//2)
        cos_cache = cos_cache.reshape(*batch_dims, seq_len, self.d_k//2)
        sin_cache = sin_cache.reshape(*batch_dims, seq_len, self.d_k//2)

        # 2 向量旋转  
        # 2.1 将x拆成两部分，形状为(..., seq_len, d_k//2, 2)
        x_pairs = rearrange(x, "... s (d_k two) -> ... s d_k two", d_k=self.d_k//2, two=2)
        # 2.2 将x_pairs拆成两部分，形状为(..., seq_len, d_k//2) 
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]
        # 2.3 将a和b分别与cos_cache和sin_cache相乘，得到(..., seq_len, d_k//2)
        a_rot = a * cos_cache - b * sin_cache
        b_rot = a * sin_cache + b * cos_cache

        # 3 维度重组
        # 这里two=2是因为einops会默认把a_rot和b_rot拼成一个张量，放在下标0的维度
        x_rot = rearrange([a_rot, b_rot], "two ... s d_k -> ... s (d_k two)", two=2)
        return x_rot.to(x.dtype)

def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    # Q: (batch_size, ..., seq_len_q, d_k)
    # K: (batch_size, ..., seq_len_k, d_k)
    # V: (batch_size, ..., seq_len_v, d_v)
    # mask: (seq_len_q, seq_len_k)
    d_k = Q.shape[-1]
    scores = einops.einsum(Q, K, "b ... q d_k, b ... k d_k -> b ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    return softmax(scores, -1) @ V

class MultiheadSelfAttention(nn.Module):
    d_model: int
    num_heads:int
    d_k: int

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # d_k = d_v = d_model/num_heads 原始论文中的配置
        self.d_k = d_model // num_heads


        # 用一个线性层将x线性变换为q,k,v,
        # qkv: (batch, ..., seq_len, 3 * num_heads * d_k)
        self.w_q = Linear(self.d_model, self.num_heads * self.d_k , device=device, dtype=dtype)
        self.w_k = Linear(self.d_model, self.num_heads * self.d_k , device=device, dtype=dtype)
        self.w_v = Linear(self.d_model, self.num_heads * self.d_k , device=device, dtype=dtype)
        self.w_o = Linear(self.num_heads * self.d_k, self.d_model, device=device, dtype=dtype)
        
    def forward(self,x:Tensor):
        # x:(batch, ..., seq_len, d_model)
        # 输出:(batch, ..., seq_len, d_model)
        seq_len = x.shape[-2]
        # 线性变换
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        # 拆分头
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        # 掩码
        mask = torch.tril(torch.ones(seq_len, seq_len,dtype=torch.bool, device=x.device),diagonal=0)
        # 缩放点积注意力
        attn = scaled_dot_product_attention(Q,K,V,mask)
        # 拼接头
        attn = rearrange(attn, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")
        return self.w_o(attn)

class MultiheadSelfAttentionWithRoPE(MultiheadSelfAttention):
    rope: RoPE
    theta: float

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10000.0,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None: 
        super().__init__(d_model, num_heads, device=device, dtype=dtype)
        self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
    
    def forward(self,x:Tensor,token_positions:Tensor):
        # x:(batch, ..., seq_len, d_model)
        # 输出:(batch, ..., seq_len, d_model)
        seq_len = x.shape[-2]
        # 线性变换
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x) 
        # 拆分头
        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)
        # 旋转
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        # 掩码
        mask = torch.tril(torch.ones(seq_len, seq_len,dtype=torch.bool, device=x.device))
        # 缩放点积注意力
        attn = scaled_dot_product_attention(Q,K,V,mask)
        # 拼接头
        attn = rearrange(attn, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")
        return self.w_o(attn)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model:int,
        num_heads:int,
        d_ff:int,
        max_seq_len:int,
        theta:float,
        weights:dict[str,Tensor],
        in_features:Float[Tensor,"batchsequence_lengthd_model"]
    ) -> None:
        """
        Args:
            d_model (int): 输入特征的维度
            num_heads (int): 多头注意力机制的头数
            d_ff (int): 前馈神经网络的中间维度
            max_seq_len (int): 最大序列长度
            theta (float): RoPE的参数
            weights (dict[str,Tensor]): 权重字典
            in_features (Float[Tensor," batch sequence_length d_model"]): 输入特征
        Returns:
            Float[Tensor," batch sequence_length d_model"]: 输出特征
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mhar = MultiheadSelfAttentionWithRoPE(d_model,num_heads,max_seq_len,theta)
        self.ffn = PositionWiseFFN(d_model,d_ff)

    def forward(
        self,
        x:Float[Tensor," batch sequence_length d_model"]
    ) -> Float[Tensor," batch sequence_length d_model"]:
        """
        Args:
            x (Float[Tensor," batch sequence_length d_model"]): 输入特征
        Returns:
            Float[Tensor," batch sequence_length d_model"]: 输出特征
        """
        token_positions = torch.arange(x.shape[-2],device=x.device)
        attn_in = self.mhar(self.norm1(x),token_positions)
        attn_out = attn_in + x
        ffn_out = self.ffn(self.norm2(attn_out))
        return ffn_out + attn_out

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size:int,
        context_length:int,
        d_model:int,
        num_layers:int,
        num_heads:int,
        d_ff:int,
        rope_theta:float,
        weights:dict[str,Tensor],
        in_indices:Int[Tensor,"batch_sizesequence_length"]
    ) -> None:
        """
        Args:
            vocab_size (int): 词汇表大小
            context_length (int): 上下文长度
            d_model (int): 模型维度
            num_layers (int): 层数
            num_heads (int): 多头注意力机制的头数
            d_ff (int): 前馈神经网络的中间维度
            rope_theta (float): RoPE的参数
            weights (dict[str,Tensor]): 权重字典
            in_indices (Int[Tensor," batch_size sequence_length"]): 输入索引
        """
        super().__init__()
        self.embedding = Embedding(vocab_size,d_model)
        self.transformer = nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta,weights,in_indices) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.linear = Linear(d_model,vocab_size)

    def forward(self,x:Int[Tensor,"batch_sizesequence_length"]) -> Float[Tensor,"batch_sizesequence_lengthvocab_size"]:
        """
        Args:
            x (Int[Tensor," batch_size sequence_length"]): 输入索引
        Returns:
            Float[Tensor," batch_size sequence_length vocab_size"]: 输出概率
        """
        x = self.embedding(x)
        for layer in self.transformer:
            x = layer(x)
        x = self.ln_final(x)
        return self.linear(x)
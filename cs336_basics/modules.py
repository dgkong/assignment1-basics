import math

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.utils import silu_activation, softmax


def scaled_dot_product_attention(
        q: Float[Tensor, "batch_size ... seq_len d_k"], 
        k: Float[Tensor, "batch_size ... seq_len d_k"], 
        v: Float[Tensor, "batch_size ... seq_len d_v"],
        mask: Bool[Tensor, "seq_len seq_len"]|None = None
) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    d_k = q.shape[-1]
    attn_scores = einsum(q, k, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        attn_scores = torch.where(mask, attn_scores, float('-inf'))
    out = einsum(softmax(attn_scores, -1), v, "... q k, ... k d_v -> ... q d_v")
    return out
    

class Linear(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            device: torch.device|None = None, 
            dtype: torch.dtype|None = None
    ):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        w = nn.init.trunc_normal_(
            torch.empty((out_features, in_features), device=device, dtype=dtype), 
            mean=0, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(w)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        out = einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
        return out


class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int, 
            embedding_dim: int,
            device: torch.device|None = None, 
            dtype: torch.dtype|None = None
    ):
        super().__init__()
        w = nn.init.trunc_normal_(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
            mean=0, std=1, a=-3, b=3
        )
        self.weight = nn.Parameter(w)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        out = self.weight[token_ids]
        return out
    

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device|None = None,
            dtype: torch.dtype|None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(reduce(x.pow(2), "... d_model -> ... 1", "mean") + self.eps)
        out = x * rms * self.weight
        return out.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device|None = None,
            dtype: torch.dtype|None = None
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        out = self.w2(silu_activation(self.w1(x)) * self.w3(x))
        return out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device|None = None      
    ):
        super().__init__()
        self.d_k = d_k
        positions = torch.arange(0, max_seq_len, device=device) 
        ang_freq = 1.0 / theta**(torch.arange(0, d_k, 2, device=device).float() / d_k) # (d_k/2,)
        angles = einsum(positions, ang_freq, "i, j -> i j") # outer prod, same as positions.unsqueeze(1) * ang_freq

        self.register_buffer("cos_cache", torch.cos(angles), persistent=False) # (max_seq_len, d_k/2)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False) # (max_seq_len, d_k/2)

    def forward(
            self, 
            x: Float[Tensor, " ... seq_len d_k"], 
            token_positions: Int[Tensor, " ... seq_len"]|None = None
    ) -> Float[Tensor, " ... seq_len d_k"]:
        if token_positions is None:
            token_positions = torch.arange(0, x.shape[-2], device=x.device)
        cos = self.cos_cache[token_positions] # (seq_len, d_k/2)
        sin = self.sin_cache[token_positions] # (seq_len, d_k/2)

        x_even = x[..., ::2] # (... seq_len, d_k/2)
        x_odd = x[..., 1::2] # (... seq_len, d_k/2)

        x_rot_even = x_even * cos - x_odd * sin # (... seq_len, d_k/2)
        x_rot_odd = x_even * sin + x_odd * cos # (... seq_len, d_k/2)

        out = torch.stack([x_rot_even, x_rot_odd], dim=-1) # (... seq_len, d_k/2, 2)
        out = rearrange(out, '... i j -> ... (i j)') # (... seq_len, d_k)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            max_seq_len: int = None,
            theta: float = None,
            device: torch.device|None = None,
            dtype: torch.dtype|None = None
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        
        self.rope = None
        if max_seq_len and theta:
            self.rope = RotaryPositionalEmbedding(theta, d_model//num_heads, max_seq_len, device)

    def forward(
            self,
            x: Float[Tensor, " ... seq_len d_model"], 
            mask: Bool[Tensor, "seq_len seq_len"]|None = None,
            token_positions: Int[Tensor, " ... seq_len"]|None = None
    ) -> Float[Tensor, " ... seq_len d_model"]:
        seq_len = x.shape[-2]
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = rearrange(q, " ... t (h d) -> ... h t d", h=self.num_heads)
        k = rearrange(k, " ... t (h d) -> ... h t d", h=self.num_heads)
        v = rearrange(v, " ... t (h d) -> ... h t d", h=self.num_heads)

        if self.rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)) # causal mask

        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = rearrange(attn, " ... h t d -> ... t (h d)", h=self.num_heads)
        out = self.output_proj(attn)
        return out


class Block(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            device: torch.device|None = None,
            dtype: torch.dtype|None = None
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        
    def forward(self, x: Float[Tensor, "batch_size seq_len d_model"]) -> Float[Tensor, "batch_size seq_len d_model"]:
        y = x + self.attn(self.ln1(x))
        out = y + self.ffn(self.ln2(y))
        return out

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device: torch.device|None = None,
            dtype: torch.dtype|None = None
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            Block(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: Int[Tensor, "batch_size seq_len"]) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        out = self.lm_head(x)
        return out

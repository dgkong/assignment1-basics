import math

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int
from torch import Tensor


def silu_activation(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


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
            token_positions: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        cos = self.cos_cache[token_positions] # (seq_len, d_k/2)
        sin = self.sin_cache[token_positions] # (seq_len, d_k/2)

        x_even = x[:, :, ::2] # (... seq_len, d_k/2)
        x_odd = x[:, :, 1::2] # (... seq_len, d_k/2)

        x_rot_even = x_even * cos - x_odd * sin # (... seq_len, d_k/2)
        x_rot_odd = x_even * sin + x_odd * cos # (... seq_len, d_k/2)

        out = torch.stack([x_rot_even, x_rot_odd], dim=-1) # (... seq_len, d_k/2, 2)
        out = rearrange(out, '... i j -> ... (i j)') # (... seq_len, d_k)
        return out

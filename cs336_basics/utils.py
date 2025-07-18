import math
from typing import Iterable

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor


def gradient_clip(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6) -> Float[Tensor, ""]:
    p_grad = [p.grad for p in params if p.grad is not None]
    if not p_grad:
        return
    total_norm = torch.sqrt(sum((g ** 2).sum() for g in p_grad))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        with torch.no_grad():
            for g in p_grad:
                g.data *= scale
    return total_norm

def get_lr_cos_schedule(curr_iter: int, max_lr: float, min_lr: float, 
                        warmup_iter: int, cos_anneal_iter: int) -> float:
    if curr_iter < warmup_iter:
        return max_lr * curr_iter / warmup_iter
    elif curr_iter <= cos_anneal_iter:
        cos_anneal = 1 + math.cos(math.pi * (curr_iter-warmup_iter) / (cos_anneal_iter-warmup_iter))
        return min_lr + 0.5 * cos_anneal * (max_lr - min_lr)
    else:
        return min_lr
    
def cross_entropy_loss(
        logits: Float[Tensor, "batch_size ... vocab_size"], 
        targets: Int[Tensor, "batch_size ..."]
    ) -> Float[Tensor, ""]:
    targets = rearrange(targets, "b ... -> (b ...)")
    logits = rearrange(logits, "b ... d -> (b ...) d")
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.logsumexp(logits, dim=-1)
    nll = -logits[torch.arange(logits.shape[0]), targets]
    loss = nll + logsumexp
    return loss.mean()

def perplexity(
        logits: Float[Tensor, "seq_len vocab_size"],
        targets: Int[Tensor, "seq_len"]
) -> Float[Tensor, ""]:
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.logsumexp(logits, dim=-1)
    nll = -logits[torch.arange(logits.shape[0]), targets]
    loss = nll + logsumexp
    return loss.mean().exp()

def silu_activation(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x = x - x.max(dim=dim, keepdim=True).values
    x_exp = x.exp()
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

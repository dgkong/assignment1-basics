import argparse
import json
import os
import time
import typing

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Int
from torch import Tensor

from cs336_basics.modules import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import (cross_entropy_loss, get_lr_cos_schedule,
                                gradient_clip)

CHECKPOINT_PATH = "cs336_basics/checkpoints"

def get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Int[Tensor, "batch_size context_len"], Int[Tensor, "batch_size context_len"]]:
    N = len(dataset)
    start_indices = torch.randint(N - context_length, (batch_size,))

    x = torch.stack([
        torch.from_numpy(dataset[i : i + context_length]).long()
        for i in start_indices
    ])
    y = torch.stack([
        torch.from_numpy(dataset[i + 1 : i + context_length + 1]).long()
        for i in start_indices
    ])

    return x.to(device), y.to(device)

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "iter": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None
) -> int:
    checkpoint = torch.load(src)
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optim"])
    return checkpoint["iter"]

def get_args():
    parser = argparse.ArgumentParser(description="Train a transformer LM")
    parser.add_argument("--config", type=str, default="cs336_basics/configs/config.json", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def get_optimizer(model: torch.nn.Module, optim_config: dict) -> torch.optim.Optimizer:
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': optim_config['weight_decay']},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print(f"num total trainable params: {num_decay_params + num_nodecay_params} parameters")
    optimizer = AdamW(optim_groups, **optim_config)
    return optimizer

def main():
    # Load Config
    args = get_args()
    with open(args.config) as f:
        config = json.load(f)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    print(f"using device: {device}, dtype: {dtype}")

    torch.manual_seed(args.seed)

    BATCH_SIZE = config['batch_size']
    CONTEXT_LENGTH = config['model']['context_length']

    MAX_STEPS = config['max_steps']
    MAX_LR = config['max_lr']
    MIN_LR = config['min_lr']
    WARMUP_ITERS = config['warmup_iters']
    MAX_NORM = config['max_l2_norm']

    SAVE_INTERVAL = config['checkpoint_interval']
    VAL_INTERVAL = config['val_interval']
    VAL_STEPS = config['val_steps']
    
    # Load Data
    TRAIN_DATASET = np.load(config['train_data'], mmap_mode='r')
    VAL_DATASET = np.load(config['val_data'], mmap_mode='r')

    # Load Model and Optimizer
    model = TransformerLM(**config['model'], device=device, dtype=dtype)
    model.to(device)
    optimizer = get_optimizer(model, config['optimizer'])

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    log_file = os.path.join(CHECKPOINT_PATH, f"log.txt")
    if args.checkpoint is None:
        global_optimizer_step = 0
        with open(log_file, "w") as f:
            pass # open for writing to clear the file
    else:
        global_optimizer_step = load_checkpoint(args.checkpoint, model, optimizer)

    def run_validation():
        print("Running validation...")
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for _ in range(VAL_STEPS):
                x, y = get_batch(VAL_DATASET, BATCH_SIZE, CONTEXT_LENGTH, device)
                logits = model(x)
                loss = cross_entropy_loss(logits, y)
                val_loss_accum += loss.item()
        model.train()
        avg_loss = val_loss_accum / VAL_STEPS
        return avg_loss

    while global_optimizer_step < MAX_STEPS:
        # CHECKPOINT
        if global_optimizer_step > 0 and global_optimizer_step % SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, global_optimizer_step, os.path.join(CHECKPOINT_PATH, f"latest_model.pt"))

        # VALIDATION
        if global_optimizer_step % VAL_INTERVAL == 0:
            val_loss = run_validation()
            print(f"step {global_optimizer_step:4d}/{MAX_STEPS} | validation loss: {val_loss:.6f}")
            with open(log_file, "a") as f:
                f.write(f"{global_optimizer_step} val {val_loss:.6f}\n")
            
        # TRAINING
        t0 = time.time()
        optimizer.zero_grad()
        tokens_processed = 0

        x, y = get_batch(TRAIN_DATASET, BATCH_SIZE, CONTEXT_LENGTH, device)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        loss.backward()
        norm = gradient_clip(model.parameters(), MAX_NORM)

        tokens_processed += x.numel()
        curr_lr = get_lr_cos_schedule(global_optimizer_step, MAX_LR, MIN_LR, WARMUP_ITERS, MAX_STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = curr_lr
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = tokens_processed / dt
        print(f"step {global_optimizer_step:4d}/{MAX_STEPS} | loss: {loss.item():.6f} | lr: {curr_lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.0f}")
        with open(log_file, "a") as f:
            f.write(f"{global_optimizer_step} train {loss.item():.6f}\n")
        global_optimizer_step += 1
        
    print("Training finished. Saving model.")
    save_checkpoint(model, optimizer, global_optimizer_step, os.path.join(CHECKPOINT_PATH, 'final_model.pt'))
    
if __name__ == "__main__":
    main()

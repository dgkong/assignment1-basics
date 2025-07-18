import json
from pathlib import Path

import torch
from jaxtyping import Int
from torch import Tensor

from cs336_basics.modules import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import get_args, load_checkpoint
from cs336_basics.utils import softmax


def decode(
        model: torch.nn.Module, 
        tokenizer: Tokenizer, 
        prompt: str, 
        max_length: int, 
        temperature: float, 
        top_p: float
) -> str:
    device = next(model.parameters()).device
    prompt = tokenizer.encode(prompt)
    while len(prompt) < max_length:
        with torch.no_grad():
            logits = model(torch.tensor(prompt, device=device)) # (seq_len, vocab_size)
            next_tok_probs = softmax(logits[-1, :], dim=-1)
            next_tok_id = torch.argmax(next_tok_probs)
            prompt.append(next_tok_id.item())
    return tokenizer.decode(prompt)

def main():
    # Load tokenizer
    BPE_PATH = (Path(__file__).resolve().parent) / "models"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(BPE_PATH / "ts_vocab.txt", BPE_PATH / "ts_merges.txt", special_tokens)

    # Load model
    args = get_args()
    with open(args.config) as f:
        config = json.load(f)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"using device: {device}, dtype: {dtype}")

    model = TransformerLM(**config['model'], device=device, dtype=dtype)
    model.to(device)

    CHECKPOINT = (Path(__file__).resolve().parent) / "checkpoints"
    # load_checkpoint(CHECKPOINT, model)
    print(decode(model, tokenizer, "The apple", max_length=32, temperature=0.7, top_p=0.9))

if __name__ == "__main__":
    main()


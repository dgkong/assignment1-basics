import json
from pathlib import Path

import torch

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
    device = model.device
    context_len = model.context_length
    eot_id = tokenizer.encode("<|endoftext|>")[0]
    prompt = tokenizer.encode(prompt)
    while len(prompt) < max_length:
        with torch.no_grad():
            if len(prompt) > context_len:
                prompt = prompt[-context_len:]
            input_ids = torch.tensor(prompt, device=device).unsqueeze(0)  # (1, seq_len)
            logits = model(input_ids)  # (1, seq_len, vocab_size)
            next_tok_probs = softmax(logits[0, -1, :] / temperature, dim=-1) # (vocab_size,)
            sorted_probs, sorted_indices = torch.sort(next_tok_probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=0)
            top_p_mask = cum_probs <= top_p
            last_included = top_p_mask.sum().item()
            top_p_mask[:last_included+1] = True # always at least 1 tok

            top_probs = sorted_probs[top_p_mask]
            top_indices = sorted_indices[top_p_mask]
            next_token_id = top_indices[torch.multinomial(top_probs, 1).item()].item()
            if next_token_id == eot_id:
                print("Ended generation because of <|eot|>.")
                break
            prompt.append(next_token_id)

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
    model = torch.compile(model, backend="aot_eager")

    # print(decode(model, tokenizer, "Once", max_length=32, temperature=0.7, top_p=0.9))
    BEST_CHECKPOINT = (Path(__file__).resolve().parent) / "out" / "checkpoints" / "lr_tune" / "batch_size_32" / "0.0009_final_model.pt"
    load_checkpoint(BEST_CHECKPOINT, model)
    print(decode(model, tokenizer, "Once", max_length=256, temperature=0.7, top_p=0.9))

if __name__ == "__main__":
    main()



import os
import random
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import regex as re
from tqdm import tqdm


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]|None = None):
        self.id_to_token = vocab
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.token_cache = {} # saves previously seen pretokens' ids
        
        self.special_tokens = None
        self.special_tokens_re = None
        if special_tokens:
            self.special_tokens = set(special_tokens)
            self.special_tokens_re = re.compile("(" + "|".join(map(re.escape, sorted(special_tokens, key=len, reverse=True))) + ")")
            for tok_str in special_tokens:
                tok = tok_str.encode("utf-8")
                if tok not in self.token_to_id:
                    ind = len(self.token_to_id)
                    self.token_to_id[tok] = ind
                    self.id_to_token[ind] = tok

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]|None = None):
        vocab = {}
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            for line in vf:
                line = line.strip()
                if not line:
                    continue
                key_str, byte_repr = line.split("\t")
                key = int(key_str)
                value = eval(byte_repr)
                vocab[key] = value
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                left, right = line.split("\t")
                left_b = eval(left)
                right_b = eval(right)
                merges.append((left_b, right_b))
        return cls(vocab, merges, special_tokens)
    
    def _encode(self, text: str) -> list[int]:
        pretok_text = []
        for match in self.pattern.finditer(text):
            pretok_text.append(match.group())
        text_encoding = []
        for pretok in pretok_text:
            if pretok in self.token_cache:
                text_encoding.extend(self.token_cache[pretok])
                continue
            
            pretok_bytes = [bytes([b]) for b in pretok.encode("utf-8")]
            if len(pretok_bytes) <= 1:
                if pretok_bytes:
                    text_encoding.append(self.token_to_id[pretok_bytes[0]])
                continue
            
            while True:
                min_rank = float('inf')
                best_pair_idx = -1
                for i in range(len(pretok_bytes) - 1):
                    pair = (pretok_bytes[i], pretok_bytes[i+1])
                    rank = self.merge_ranks.get(pair)
                    if rank is not None and rank < min_rank:
                        min_rank = rank
                        best_pair_idx = i
                if best_pair_idx == -1:
                    break
                merged_token = pretok_bytes[best_pair_idx] + pretok_bytes[best_pair_idx + 1]
                pretok_bytes = pretok_bytes[:best_pair_idx] + [merged_token] + pretok_bytes[best_pair_idx + 2:]

            encoding = [self.token_to_id[b] for b in pretok_bytes]
            self.token_cache[pretok] = encoding
            text_encoding.extend(encoding)

        return text_encoding

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            text_chunks = self.special_tokens_re.split(text)
        else:
            return self._encode(text)

        text_encoding = []
        for chunk in text_chunks:
            if chunk in self.special_tokens:
                text_encoding.append(self.token_to_id[chunk.encode("utf-8")])
            else:
                text_encoding.extend(self._encode(chunk))
        return text_encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens = b''.join([self.id_to_token[t] for t in ids])
        return tokens.decode("utf-8", errors="replace")
    
    def encode_file(self, f: Iterable[str], chunk_size: int = 10 * 2**20) -> Iterator[int]:
        leftover = ""
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            block = leftover + block
            last_eot_end = -1
            for m in self.special_tokens_re.finditer(block):
                last_eot_end = m.end()
            if last_eot_end == -1:
                leftover = block
            else:
                yield from self.encode(block[:last_eot_end])
                leftover = block[last_eot_end:]
        if leftover:
            yield from self.encode(leftover)
    
def sample_bytes(data_path, size: int) -> bytes:
    data_size = os.path.getsize(data_path)
    start = random.randint(0, data_size - size)
    with open(data_path, "rb") as f:
        f.seek(start)
        return f.read(size)

def sample_compression_ratios(data_path, tokenizer):
    sample_size = 10 * 2**10
    compression_ratios = []
    for i in tqdm(range(10)):
        print(i)
        ts_sample = sample_bytes(data_path, sample_size)
        ts_sample_text = ts_sample.decode("utf-8")
        encoded_ids = tokenizer.encode(ts_sample_text)
        compression_ratios.append(len(ts_sample) / len(encoded_ids))
    print(f"average compression ratio:\t{sum(compression_ratios)/len(compression_ratios)}")
    return compression_ratios

def tokenize_data(data_path, tokenizer, save_file=None):
    start = time.time()
    with open(data_path, "r", encoding="utf-8") as f:
        tokens = list(tokenizer.encode_file(f))
    elapsed = time.time() - start

    print(f"total time:\t{elapsed:.4f} seconds")
    data_size = os.path.getsize(data_path)
    print(f"data size:\t{data_size / 1e6} MB")
    print(f"throughput:\t{data_size / elapsed / 1e6:.4f} MB per second")

    if save_file:
        np.save(save_file, np.array(tokens, dtype=np.uint16))


if __name__ == "__main__":
    DATA_PATH = (Path(__file__).resolve().parent.parent) / "data"
    BPE_PATH = (Path(__file__).resolve().parent) / "models"
    TS_TRAIN = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    TS_VAL = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
    OWT_TRAIN = DATA_PATH / "owt_train.txt"
    OWT_VAL = DATA_PATH / "owt_valid.txt"

    random.seed(42)
    special_tokens = ["<|endoftext|>"]
    # ts_tokenizer = Tokenizer.from_files(BPE_PATH / "ts_vocab.txt", BPE_PATH / "ts_merges.txt", special_tokens)
    owt_tokenizer = Tokenizer.from_files(BPE_PATH / "owt_vocab.txt", BPE_PATH / "owt_merges.txt", special_tokens)

    # ts_compression_ratios = sample_compression_ratios(TS_TRAIN, ts_tokenizer)
    # owt_compression_ratios = sample_compression_ratios(OWT_TRAIN, owt_tokenizer)

    # tokenize_data(TS_VAL, ts_tokenizer, DATA_PATH / "ts_val_tokenized.npy")
    # tokenize_data(TS_TRAIN, ts_tokenizer, DATA_PATH / "ts_train_tokenized.npy")

    # tokenize_data(OWT_VAL, owt_tokenizer)
    tokenize_data(OWT_TRAIN, owt_tokenizer, DATA_PATH / "owt_train_tokenized.npy")

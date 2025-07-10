
from typing import Iterable, Iterator

import regex as re


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]=None):
        self.id_to_token = vocab
        self.merges = merges
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
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
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]=None):
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
            pretok_text.append(tuple(bytes([b]) for b in match.group().encode("utf-8")))
        for i, pretok in enumerate(pretok_text):
            new_pretok = []
            for merge in self.merges:
                j = 0
                while j < len(pretok):
                    if j < len(pretok) - 1 and (pretok[j], pretok[j+1]) == merge:
                        new_pretok.append(b''.join(merge))
                        j += 1
                    else:
                        new_pretok.append(pretok[j])
                    j += 1
                pretok = tuple(new_pretok)
                new_pretok = []
            pretok_text[i] = pretok

        text_encoding = []
        for pretok in pretok_text:
            for tok in pretok:
                text_encoding.append(self.token_to_id[tok])
        return text_encoding


    def encode(self, text: str) -> list[int]:
        if self.special_tokens_re:
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

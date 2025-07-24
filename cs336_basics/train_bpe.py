import heapq
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import BinaryIO

import regex as re
from tqdm import tqdm

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class BPHeapItem:
    """
    Wrapper class for byte bp heap items. 
    Items with higher count and lexicographical values will have priority.
    """
    def __init__(self, key, count):
        self.key = key
        self.count = count

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        return self.key > other.key


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens_re: re.Pattern
) -> Counter[tuple[bytes, ...]]:
    
    pretok_freqs = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunk_docs = special_tokens_re.split(chunk)
        for doc in chunk_docs:
            for match in PAT.finditer(doc):
                pretok = tuple(bytes([b]) for b in match.group().encode("utf-8"))
                pretok_freqs[pretok] += 1
    return pretok_freqs

def pretokenize(
    input_path: str,
    special_tokens: list[str],
    num_processes: int
) -> tuple[dict[tuple[bytes, ...], int], dict[int, tuple[bytes, ...]], dict[tuple[bytes, ...], int]]:
    
    pretok_freqs = Counter()
    special_tokens_re = re.compile("|".join(list(map(re.escape, special_tokens))))
    print(f"Running pre-tokenization with {num_processes} processes...")
    if num_processes == 1:
        with open(input_path, "rb") as f:
            start, end = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk_docs = special_tokens_re.split(chunk)
            for doc in tqdm(chunk_docs, total=len(chunk_docs), desc="Docs", unit="doc"):
                for match in PAT.finditer(doc):
                    pretok = tuple(bytes([b]) for b in match.group().encode("utf-8"))
                    pretok_freqs[pretok] += 1
    else:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, 8*num_processes, "<|endoftext|>".encode("utf-8"))
        worker_args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            worker_args.append((input_path, start, end, special_tokens_re))
        with mp.Pool(processes=num_processes) as pool:
            freq_counters = pool.starmap(process_chunk, worker_args)
        for counter in freq_counters:
            pretok_freqs.update(counter)
    print(f"Finished pre-tokenization!")
    pretok_to_index = {}
    index_to_pretok = {}
    for i, pretok in enumerate(pretok_freqs.keys()):
        pretok_to_index[pretok] = i
        index_to_pretok[i] = pretok

    return dict(pretok_freqs), pretok_to_index, index_to_pretok

def get_bp_freqs(
    pretok_freqs: dict[tuple[bytes, ...], int],
    pretok_to_index: dict[tuple[bytes, ...], int]
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[int]]]:
    
    bp_freqs, bp_pretok_map = defaultdict(int), defaultdict(set)
    print(f"Counting byte-pair frequencies...")
    for pretok, freq in pretok_freqs.items():
        if len(pretok) == 1:
            continue
        for i in range(len(pretok)-1):
            bp = (pretok[i], pretok[i+1])
            bp_freqs[bp] += freq
            bp_pretok_map[bp].add(pretok_to_index[pretok])

    return dict(bp_freqs), dict(bp_pretok_map)

def update_merged_bp(
    bp_to_merge: tuple[bytes, bytes],
    bp_freqs: dict[tuple[bytes, bytes], int],
    bp_pretok_map: dict[tuple[bytes, bytes], set[int]],
    bp_heap: list[BPHeapItem],
    pretok_freqs: dict[tuple[bytes, ...], int],
    index_to_pretok: dict[int, tuple[bytes, ...]]
):
    merged_bp = b''.join(bp_to_merge)
    updated_bps = set()
    included_pretoks = bp_pretok_map[bp_to_merge]
    for ind in included_pretoks:
        pretok = index_to_pretok[ind]
        freq = pretok_freqs[pretok]

        # decrement old pretoken bps
        for i in range(len(pretok) - 1):
            bp = (pretok[i], pretok[i+1])
            bp_freqs[bp] -= freq
            updated_bps.add(bp)

        new_pretok = []
        i = 0
        while i < len(pretok):
            if i < len(pretok) - 1 and (pretok[i], pretok[i+1]) == bp_to_merge:
                new_pretok.append(merged_bp)
                i += 1
            else:
                new_pretok.append(pretok[i])
            i += 1
        new_pretok = tuple(new_pretok)

        del pretok_freqs[pretok]
        pretok_freqs[new_pretok] = freq
        index_to_pretok[ind] = new_pretok

        # increment new pretok bps
        for i in range(len(new_pretok) - 1):
            bp = (new_pretok[i], new_pretok[i+1])
            bp_freqs[bp] = bp_freqs.get(bp, 0) + freq
            if bp not in bp_pretok_map:
                bp_pretok_map[bp] = set()
            bp_pretok_map[bp].add(ind)
            updated_bps.add(bp)
        
    del bp_freqs[bp_to_merge]
    del bp_pretok_map[bp_to_merge]
    updated_bps.remove(bp_to_merge)

    for bp in updated_bps:
        heapq.heappush(bp_heap, BPHeapItem(bp, bp_freqs[bp]))
        
def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str],
    num_processes: int = max(mp.cpu_count()-1, 1)
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pretok_freqs, pretok_to_index, index_to_pretok = pretokenize(input_path, special_tokens, num_processes)
    bp_freqs, bp_pretok_map = get_bp_freqs(pretok_freqs, pretok_to_index)

    print(f"Merging byte-pair encodings...")
    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")
    merges = []
    bp_heap = [BPHeapItem(bp, freq) for bp, freq in bp_freqs.items()]
    heapq.heapify(bp_heap)

    target_merges = vocab_size - len(vocab)
    pbar = tqdm(total=target_merges, desc="BPE merges", unit="merge")
    while len(vocab) < vocab_size:
        if not bp_heap:
            break
        bp_heap_item = heapq.heappop(bp_heap)
        bp_to_merge, count = bp_heap_item.key, bp_heap_item.count
        if bp_to_merge not in bp_freqs or bp_freqs[bp_to_merge] != count:
            continue

        update_merged_bp(bp_to_merge, bp_freqs, bp_pretok_map, bp_heap, pretok_freqs, index_to_pretok)
        
        vocab[len(vocab)] = b''.join(bp_to_merge)
        merges.append(bp_to_merge)
        pbar.update()
    pbar.close()
    return vocab, merges

def save_bpe(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_name: str
):
    print(f"Saving BPE vocab and merges...")
    with open("./bpe_models/" + output_name + "_vocab.txt", "w", encoding="utf-8") as f:
        for index, token_bytes in vocab.items():
            f.write(f"{index}\t{repr(token_bytes)}\n")

    with open("./bpe_models/" + output_name + "_merges.txt", "w", encoding="utf-8") as f:
        for token_a, token_b in merges:
            f.write(f"{repr(token_a)}\t{repr(token_b)}\n")

if __name__ == "__main__":
    DATA_PATH = (Path(__file__).resolve().parent.parent) / "data"

    # TinyStories
    input_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"]
    )
    # save_bpe(vocab, merges, "ts")

    # OpenWebText
    # input_path = DATA_PATH / "owt_train.txt"
    # vocab, merges = train_bpe(
    #     input_path=input_path,
    #     vocab_size=32000,
    #     special_tokens=["<|endoftext|>"]
    # )
    # save_bpe(vocab, merges, "owt")
    

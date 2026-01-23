import heapq_max
import multiprocessing
from .find_chunk_boundaries import find_chunk_boundaries
from .pretokenization import pretokenize
from .utils import DEFAULT_PROCESS_COUNT, DEFAULT_CHUNK_NUM


def _pretokenize_worker(args: tuple[str, int, int, list[str]]) -> dict[tuple[bytes], int]:
    """
    Worker function for parallel pretokenization.
    Must be at module level to be picklable.
    """
    input_path, start, end, special_tokens = args
    return pretokenize(
        file_dir=input_path,
        start=start,
        end=end,
        special_tokens=special_tokens
    )

def _join_pretokens(
    pretoken_dicts: list[dict[tuple[bytes], int]]
) -> dict[tuple[bytes], int]:
    """
    Combine multiple pretoken frequency dictionaries into one.
    """
    combined = {}
    for pt in pretoken_dicts:
        for token, count in pt.items():
            if token in combined:
                combined[token] += count
            else:
                combined[token] = count
    return combined
    

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = DEFAULT_PROCESS_COUNT,
    chunk_num: int = DEFAULT_CHUNK_NUM
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the input text file.
    Returns the vocabulary and the list of BPE merges.
    """
    # Step 1: Find chunk boundaries for parallel processing
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_processes,
            split_special_token=b"<|endoftext|>"
        )
    
    # Step 2: process chunks in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        pretoken_freq_dicts = pool.map(
            _pretokenize_worker,
            [
                (input_path, chunk_boundaries[i], chunk_boundaries[i + 1], special_tokens)
                for i in range(len(chunk_boundaries) - 1)
            ]
        )
    
    # Step 3: Combine the frequency dictionaries from all chunks
    combined_pretoken_freq = _join_pretokens(pretoken_freq_dicts)

    # Step 4: Train BPE on the combined frequency dictionary
    # initial vocab include special tokens and 256 byte tokens
    vocab = {i: token.encode('utf-8') for i, token in enumerate(special_tokens)}
    vocab.update({i + len(special_tokens): bytes([i]) for i in range(256)})
    merges: list[tuple[bytes, bytes]] = []
    pair_freq: dict[tuple[bytes, bytes], int] = {}

    for pretoken, freq in combined_pretoken_freq.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            if pair in pair_freq:
                pair_freq[pair] += freq
            else:
                pair_freq[pair] = freq

    max_heap = [(freq, pair) for pair, freq in pair_freq.items()]
    heapq_max.heapify_max(max_heap)

    while len(vocab) < vocab_size and max_heap:
        while max_heap:
            freq, best_pair = max_heap[0]
            true_freq = pair_freq.get(best_pair, 0)
            if freq == true_freq:
                break
            heapq_max.heappop_max(max_heap)
        heapq_max.heappop_max(max_heap)
        a, b = best_pair
        new_token = a + b
        vocab[len(vocab)] = new_token
        merges.append([a, b])

        new_combined_pretoken_freq = {}
        for pretoken, freq in combined_pretoken_freq.items():
            new_pretoken = []
            i = 0
            n = len(pretoken)
            while i < n:
                if i < n - 1 and (pretoken[i], pretoken[i + 1]) == best_pair:
                    to_be_reduced = set()
                    to_be_increased = set()
                    to_be_reduced.add((pretoken[i], pretoken[i + 1]))
                    to_be_reduced.add((pretoken[i - 1], pretoken[i])) if i - 1 >= 0 else None
                    to_be_reduced.add((pretoken[i + 1], pretoken[i + 2])) if i + 2 < n else None
                    to_be_increased.add((pretoken[i - 1], new_token)) if i - 1 >= 0 else None
                    to_be_increased.add((new_token, pretoken[i + 2])) if i + 2 < n else None

                    for pair in to_be_reduced:
                        if pair in pair_freq:
                            pair_freq[pair] -= freq
                            if pair_freq[pair] <= 0:
                                del pair_freq[pair]
                            heapq_max.heappush_max(max_heap, (pair_freq.get(pair, 0), pair))
                    for pair in to_be_increased:
                        if pair in pair_freq:
                            pair_freq[pair] += freq
                        else:
                            pair_freq[pair] = freq
                        heapq_max.heappush_max(max_heap, (pair_freq[pair], pair))

                    new_pretoken.append(new_token)
                    i += 2
                else:
                    new_pretoken.append(pretoken[i])
                    i += 1
            new_pretoken_tuple = tuple(new_pretoken)
            if new_pretoken_tuple in new_combined_pretoken_freq:
                new_combined_pretoken_freq[new_pretoken_tuple] += freq
            else:
                new_combined_pretoken_freq[new_pretoken_tuple] = freq
        combined_pretoken_freq = new_combined_pretoken_freq

    return vocab, merges
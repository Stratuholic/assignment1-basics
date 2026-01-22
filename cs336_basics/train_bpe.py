import heapq
import multiprocessing
from find_chunk_boundaries import find_chunk_boundaries
from pretokenization import pretokenize, join_pretokens
from utils import DEFAULT_PROCESS_COUNT, DEFAULT_CHUNK_NUM


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


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = DEFAULT_PROCESS_COUNT,
    chunk_num: int = DEFAULT_CHUNK_NUM
): # -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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
    combined_pretoken_freq = join_pretokens(pretoken_freq_dicts)

    # Step 4: Train BPE on the combined frequency dictionary
    ### create a max heap based on frequency

        
    return combined_pretoken_freq
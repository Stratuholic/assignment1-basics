import regex as re


def pretokenize(
    file_dir: str,
    start: int,
    end: int,
    special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    """
    Pretokenize the input text wordwise into a frequency dictionary of byte sequences.
    Special tokens are treated as indivisible units.
    """
    # use the provided regex pattern for tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Split text by special tokens to handle them separately
    with open(file_dir, "r", encoding="utf-8") as f:
        f.seek(start)
        text = f.read(end - start)
        token_freq_dicts = []
        for separated_text in re.split(
            f"({'|'.join(special_tokens)})",
            text
        ):
            token_freq_dict = {}
            for pretoken in re.finditer(PAT, separated_text):
                pretoken_str = pretoken.group(0)
                pretoken_bytes = pretoken_str.encode("utf-8")
                # pretoken as a byte-by-byte tuple
                pretoken_tuple = tuple(bytes([b]) for b in pretoken_bytes)

                if pretoken_tuple in token_freq_dict:
                    token_freq_dict[pretoken_tuple] += 1
                else:
                    token_freq_dict[pretoken_tuple] = 1
            token_freq_dicts.append(token_freq_dict)
        
    joined_dict = join_pretokens(token_freq_dicts)
    return joined_dict

def join_pretokens(
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
    
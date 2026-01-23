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
    # PAT = r"\s+"

    # Split text by special tokens to handle them separately
    token_freq_dict = {}
    with open(file_dir, "rb") as f:
        f.seek(start)
        text = f.read(end - start)
        for separated_text in re.split(
            f"{'|'.join(re.escape(special_token) for special_token in special_tokens)}",
            text.decode("utf-8")
        ):
            for pretoken in re.finditer(PAT, separated_text):
                pretoken_str = pretoken.group(0)
                # pretoken_str = pretoken
                pretoken_bytes = pretoken_str.encode("utf-8")
                # pretoken as a byte-by-byte tuple
                pretoken_tuple = tuple(bytes([b]) for b in pretoken_bytes)

                if pretoken_tuple in token_freq_dict:
                    token_freq_dict[pretoken_tuple] += 1
                else:
                    token_freq_dict[pretoken_tuple] = 1

    return token_freq_dict
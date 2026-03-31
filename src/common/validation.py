from __future__ import annotations

from typing import Iterable


def validate_sequence(sequence: Iterable[int], problem_size: int) -> tuple[bool, str]:
    seq = list(sequence)
    if len(seq) != problem_size:
        return False, f"expected length {problem_size}, got {len(seq)}"
    if any(node < 0 or node >= problem_size for node in seq):
        return False, f"sequence contains indices outside 0..{problem_size - 1}"
    if len(set(seq)) != problem_size:
        return False, "sequence contains duplicates or omissions"
    return True, ""


def assert_valid_sequence(sequence: Iterable[int], problem_size: int) -> None:
    is_valid, reason = validate_sequence(sequence, problem_size)
    if not is_valid:
        raise ValueError(f"invalid sequence: {reason}")


def batch_validate_sequences(sequences: Iterable[Iterable[int]], problem_size: int) -> None:
    for sequence in sequences:
        assert_valid_sequence(sequence, problem_size)


def deduplicate_sequences(sequences: Iterable[Iterable[int]]) -> list[list[int]]:
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for sequence in sequences:
        key = tuple(sequence)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(sequence))
    return unique


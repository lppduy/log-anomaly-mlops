"""Train/test split cho feature pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np


def train_test_split_logs(
    logs: list[dict[str, Any]],
    *,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Random split logs thành train/test.

    Returns:
        train_logs, test_logs, train_indices, test_indices
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    n = len(logs)
    rng = np.random.default_rng(random_seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    test_count = max(1, int(n * test_size))
    test_indices = np.sort(indices[:test_count])
    train_indices = np.sort(indices[test_count:])

    train_logs = [logs[i] for i in train_indices]
    test_logs = [logs[i] for i in test_indices]
    return train_logs, test_logs, train_indices, test_indices

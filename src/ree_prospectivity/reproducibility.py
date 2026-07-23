"""Utilities for controlled random behavior."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for one versioned runtime."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed NumPy and Python from the worker seed assigned by PyTorch."""

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

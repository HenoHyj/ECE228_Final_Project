"""Reproducibility helper. Call once at the top of every training script."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Must be set before CUDA initializes; seed_everything is called at the top
    # of every training script, before the first CUDA op, so this takes effect.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: some ops (e.g. cuDNN LSTM backward) lack deterministic kernels;
    # warn instead of hard-failing so AMP/LSTM training still runs.
    torch.use_deterministic_algorithms(True, warn_only=True)

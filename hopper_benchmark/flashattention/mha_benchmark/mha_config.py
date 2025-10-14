from dataclasses import dataclass

import torch


@dataclass
class MHAConfig:
    """Configuration for MHA."""
    batch: int
    seq_len: int
    heads: int
    dim: int           # per-head dimension (Hd)
    causal: bool
    dtype: torch.dtype = torch.float16
    dropout_p: float = 0.0


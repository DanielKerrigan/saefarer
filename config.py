"""Configuration for SAE and ActivationsStore."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    """Configuration class for SAE and ActivationsStore."""

    d_sae: int
    d_in: int
    k: int
    hidden_state_index: int
    normalize: bool
    lm_sequence_length: int
    lm_batch_size_sequences: int
    n_batches_in_store: int
    sae_batch_size_tokens: int
    prepend_bos_token: bool
    device: Literal["cpu", "cuda"]
    dtype: Literal["float16", "bfloat16", "float32", "float64"]

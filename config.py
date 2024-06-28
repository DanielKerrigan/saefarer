"""Configuration for SAE and ActivationsStore."""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Config:
    """Configuration class for SAE and ActivationsStore."""

    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float16", "bfloat16", "float32", "float64"] = "float32"
    # dimensions
    d_sae: int = 128
    d_in: int = 64
    # loss
    k: int = 4
    aux_k: int = 32
    aux_k_coef: int = 1 / 32
    dead_tokens_threshold: int = 10_000_000
    dead_steps_threshold: Optional[int] = None
    hidden_state_index: int = -1
    # activation normalization
    normalize: bool = False
    # batch sizes
    lm_sequence_length: int = 256
    lm_batch_size_sequences: int = 32
    n_batches_in_store: int = 20
    sae_batch_size_tokens: int = 4096
    # tokenization
    prepend_bos_token: bool = True
    # adam
    lr: int = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: int = 6.25e-10

    def __post_init__(self):
        if self.dead_steps_threshold is None:
            self.dead_steps_threshold = (
                self.dead_tokens_threshold / self.sae_batch_size_tokens
            )

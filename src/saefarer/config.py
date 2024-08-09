"""Configuration for SAE and ActivationsStore."""

from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass
class Config:
    """Configuration class for SAE and ActivationsStore."""

    device: Literal["cpu", "mps", "cuda"] = "cpu"
    dtype: Literal["float16", "bfloat16", "float32", "float64"] = "float32"
    # dataset
    dataset_column: str = "input_ids"
    # dimensions
    d_in: int = 64
    expansion_factor: int = 4
    d_sae: int = field(init=False)
    # loss
    k: int = 4
    aux_k: int = 32
    aux_k_coef: float = 1 / 32
    dead_tokens_threshold: int = 10_000_000
    dead_steps_threshold: int = field(init=False)
    hidden_state_index: int = -1
    # activation normalization
    normalize: bool = False
    # batch sizes
    model_sequence_length: int = 256
    model_batch_size_sequences: int = 32
    n_batches_in_store: int = 20
    sae_batch_size_tokens: int = 4096
    # tokenization
    prepend_bos_token: bool = True
    # adam
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 6.25e-10
    # training
    total_training_tokens: int = 100_000_000
    total_training_batches: int = field(init=False)
    # logging
    logger: Literal["local", "wandb"] = "wandb"
    log_batch_freq: int = 1000
    wandb_project: Union[str, None] = None
    wandb_group: Union[str, None] = None
    wandb_name: Union[str, None] = None
    wandb_notes: Union[str, None] = None

    def __post_init__(self):
        self.dead_steps_threshold = (
            self.dead_tokens_threshold // self.sae_batch_size_tokens
        )

        self.total_training_batches = (
            self.total_training_tokens // self.sae_batch_size_tokens
        )

        self.d_sae = self.d_in * self.expansion_factor

"""Configuration for SAE analysis."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    pass


@dataclass
class AnalysisConfig:
    """Configuration class for analyzing SAEs."""

    # no default

    # dataset
    labels: list[str]

    # default

    # device
    device: 'Literal["cpu", "mps", "cuda", "xpu", "xla"]' = "cuda"
    # dataset
    tokens_column: str = "input_ids"
    attn_mask_column: str = "attention_mask"
    label_column: str = "label"
    # batch sizes
    model_batch_size_sequences: int = 32
    model_sequence_length: int = 128
    feature_batch_size: int = 256
    # analysis
    total_analysis_tokens: int = 10_000_000
    total_analysis_sequences: int = field(init=False)
    feature_indices: list[int] = field(default_factory=list)
    n_activation_bins: int = 32
    # ui
    n_example_sequences: int = 10
    n_context_tokens: int = 5
    n_sequence_intervals: int = 8
    extra_token_columns: list[str | tuple[str, Callable[[Any], str]]] = field(
        default_factory=list
    )
    extra_sequence_columns: list[str | tuple[str, Callable[[Any], str]]] = field(
        default_factory=list
    )
    # logging
    show_progress: bool = True

    def __post_init__(self):
        self.total_analysis_sequences = (
            self.total_analysis_tokens // self.model_sequence_length
        )

    @classmethod
    def from_dict(cls, data: dict):
        data_copy = data.copy()
        data_copy.pop("total_analysis_sequences")
        return cls(**data_copy)

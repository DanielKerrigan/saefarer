"""Configuration for Widget."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class WidgetConfig:
    """Configuration class for the widget."""

    height: int = 600
    base_font_size: int = 16
    n_table_rows: int = 10
    default_min_act_instances: int | None = 32
    default_min_act_rate: float | None = None
    # only for inference
    device: 'Literal["cpu", "mps", "cuda", "xpu", "xla"]' = "cuda"

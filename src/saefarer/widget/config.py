"""Configuration for Widget."""

from dataclasses import dataclass


@dataclass
class WidgetConfig:
    """Configuration class for the widget."""

    height: int = 600
    base_font_size: int = 16
    n_table_rows: int = 10

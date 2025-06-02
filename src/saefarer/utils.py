"""Utility functions."""

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

if TYPE_CHECKING:
    import numpy.typing as npt


def get_default_device() -> 'Literal["cpu", "mps", "cuda"]':
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"


def top_k_indices_values(
    x: torch.Tensor, k: int, largest: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a 2D matrix x, return the row and column indices of
    the k largest or smallest values."""
    # This code is adapted from sae-vis.
    top = x.flatten().topk(k=k, largest=largest)
    indices = top.indices
    rows = indices // x.size(1)
    cols = indices % x.size(1)
    return torch.stack((rows, cols), dim=1), top.values


def freedman_diaconis_np(
    x: "npt.NDArray", x_range: tuple[float, float] | None = None
) -> int:
    """Freedman Diaconis Estimator for determining
    the number of bins in a histogram."""
    iqr = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    bin_width = 2 * iqr / np.cbrt(x.size)

    if bin_width == 0:
        return 1

    diff = x.max() - x.min() if x_range is None else x_range[1] - x_range[0]

    n_bins = diff / bin_width
    return int(np.ceil(n_bins))

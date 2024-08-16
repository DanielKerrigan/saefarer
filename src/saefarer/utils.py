"""Utility functions."""

from typing import Tuple

import numpy as np
import torch


def top_k_indices(x: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    """Given a 2D matrix x, return the row and column indices of
    the k largest or smallest values."""
    indices = x.flatten().topk(k=k, largest=largest).indices
    rows = indices // x.size(1)
    cols = indices % x.size(1)
    return torch.stack((rows, cols), dim=1)


def torch_histogram(xs: torch.Tensor, bins: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Like torch.histogram, but works with cuda
    # https://github.com/pytorch/pytorch/issues/69519#issuecomment-1183866843
    min, max = xs.min().item(), xs.max().item()
    counts = torch.histc(xs, bins, min=min, max=max)
    boundaries = torch.linspace(min, max, bins + 1)
    return counts, boundaries


def freedman_diaconis_torch(x: torch.Tensor) -> int:
    """Freedman Diaconis Estimator for determining
    the number of bins in a histogram."""
    iqr = torch.quantile(x, 0.75) - torch.quantile(x, 0.25)
    bin_width = 2 * iqr / np.cbrt(x.numel())

    if bin_width == 0:
        return 1

    n_bins = (x.max() - x.min()) / bin_width
    return int(np.ceil(n_bins.item()))


def freedman_diaconis_np(x: np.ndarray) -> int:
    """Freedman Diaconis Estimator for determining
    the number of bins in a histogram."""
    iqr = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    bin_width = 2 * iqr / np.cbrt(x.size)

    if bin_width == 0:
        return 1

    n_bins = (x.max() - x.min()) / bin_width
    return int(np.ceil(n_bins))

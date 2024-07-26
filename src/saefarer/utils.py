"""Utility functions."""

import numpy as np
import torch


def top_k_indices(x: torch.Tensor, k: int, largest: bool = True) -> torch.Tensor:
    """Given a 2D matrix x, return the row and column indices of
    the k largest or smallest values."""
    indices = x.flatten().topk(k=k, largest=largest).indices
    rows = indices // x.size(1)
    cols = indices % x.size(1)
    return torch.stack((rows, cols), dim=1)


def freedman_diaconis(x: torch.Tensor) -> int:
    """Freedman Diaconis Estimator for determining
    the number of bins in a histogram."""
    iqr = torch.quantile(x, 0.75) - torch.quantile(x, 0.25)
    bin_width = 2 * iqr / np.cbrt(x.numel())

    if bin_width == 0:
        return 1

    n_bins = (x.max() - x.min()) / bin_width
    return int(np.ceil(n_bins.item()))

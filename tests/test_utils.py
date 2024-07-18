import numpy as np
import torch

from sparse_autoencoder.utils import freedman_diaconis


def test_freedman_diaconis():
    xt = torch.rand(1000)
    xn = xt.numpy()

    expected_n_bins = np.histogram_bin_edges(xn, bins="fd").shape[0] - 1

    assert freedman_diaconis(xt) == expected_n_bins

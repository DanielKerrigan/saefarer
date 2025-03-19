import numpy as np

from saefarer.utils import freedman_diaconis_np


def test_freedman_diaconis():
    rng = np.random.default_rng()
    x = rng.random(1000)

    expected_n_bins = np.histogram_bin_edges(x, bins="fd").shape[0] - 1

    assert freedman_diaconis_np(x) == expected_n_bins

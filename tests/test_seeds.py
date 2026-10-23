import numpy as np

from tvd.utils import set_seed


def test_set_seed_is_deterministic():
    set_seed(123)
    first = np.random.rand(4)
    set_seed(123)
    second = np.random.rand(4)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_differ():
    set_seed(1)
    a = np.random.rand(4)
    set_seed(2)
    b = np.random.rand(4)
    assert not np.array_equal(a, b)

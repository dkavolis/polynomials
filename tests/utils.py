import pytest

import numpy as np


def assert_allclose(a, b, rel=1e-6, atol=1e-12, nan_ok: bool = False):
    a = np.asarray(a)
    b = np.array(b)
    if a.dtype != np.dtype(object) and b.dtype != np.dtype(object):
        return np.testing.assert_allclose(a, b, rel, atol, nan_ok)

    assert a.size == b.size

    for i, (x, y) in enumerate(zip(a, b)):
        assert x == pytest.approx(y, rel, atol, nan_ok)

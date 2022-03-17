#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import numpy as np

from polynomials.hints import ArrayFloat


def assert_allclose(
    a: ArrayFloat,
    b: ArrayFloat,
    rel: float = 1e-6,
    atol: float = 1e-12,
    nan_ok: bool = False,
) -> None:
    a = np.asarray(a)
    b = np.array(b)
    if a.dtype != np.dtype(object) and b.dtype != np.dtype(object):
        return np.testing.assert_allclose(a, b, rel, atol, nan_ok)

    assert a.size == b.size

    for x, y in zip(a, b):
        assert x == pytest.approx(y, rel, atol, nan_ok)

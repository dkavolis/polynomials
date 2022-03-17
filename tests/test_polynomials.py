#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Iterable, Optional, Tuple, Union, Type

import pytest

import numpy as np
from polynomials.hints import Polynomial
import utils

from polynomials import (
    Chebyshev,
    Hermite,
    Laguerre,
    Legendre,
    LegendreStieltjes,
    ArrayFloat,
)


class Expected:
    def __init__(
        self,
        y: Iterable[float],
        domain: Tuple[float, float],
        prime: Iterable[float] = None,
        zeros: Iterable[float] = None,
        weights: Iterable[float] = None,
        abscissa: Iterable[float] = None,
        x: Union[float, Iterable[float]] = 0.4,
        order: int = 3,
    ):
        self.order = order
        self.y: ArrayFloat = np.asarray(y)
        self.x = np.ones_like(self.y) * np.asarray(x)
        self.domain = domain
        self.prime: Optional[ArrayFloat] = None if prime is None else np.asarray(prime)
        self.zeros: Optional[ArrayFloat] = None if zeros is None else np.asarray(zeros)
        self.weights: Optional[ArrayFloat] = (
            None if weights is None else np.asarray(weights)
        )
        self.abscissa: Optional[ArrayFloat] = (
            None if abscissa is None else np.asarray(abscissa)
        )


EXPECTED = {
    Legendre: Expected(
        y=[-0.44] * 3,
        domain=(-1, 1),
        prime=[-0.3] * 3,
        zeros=[-0.774596669, 0, 0.774596669],
        weights=[0.347854845, 0.652145155, 0.652145155, 0.347854845],
        abscissa=[-0.861136311, -0.339981044, 0.339981044, 0.861136311],
    ),
    Hermite: Expected(y=[-4.288] * 3, domain=(-np.inf, np.inf)),
    Chebyshev: Expected(
        y=[-0.944] * 3,
        domain=(-1, 1),
        prime=[-1.08] * 3,
        zeros=[-0.866025404, 0, 0.866025404],
        weights=[0.785398163] * 4,
        abscissa=[-0.923879533, -0.382683432, 0.382683432, 0.923879533],
    ),
    Laguerre: Expected(y=[0.029333333] * 3, domain=(0, 1)),
    LegendreStieltjes: Expected(
        y=[-0.697143] * 3,
        domain=(-1, 1),
        prime=[-0.942857] * 3,
        zeros=[-0.925820100, 0, 0.925820100],
    ),
}


@pytest.fixture(scope="module", params=list(EXPECTED.keys()))
def expectation(request: pytest.FixtureRequest) -> Tuple[Polynomial, Expected]:
    p: Type[Polynomial] = request.param  # type: ignore # really as SubRequest
    return p(EXPECTED[p].order), EXPECTED[p]


def uses_fixture(f: Callable[[Polynomial, Expected], None]):
    def wrapper(expectation: Tuple[Polynomial, Expected]):
        f(*expectation)

    return wrapper


@uses_fixture
def test_properties(polynomial: Polynomial, expected: Expected):
    assert polynomial.order == expected.order

    copy = type(polynomial)(polynomial)
    assert copy is not polynomial
    assert copy.order == polynomial.order
    copy.order = 1
    assert copy.order == 1

    utils.assert_allclose(type(polynomial).domain, expected.domain)


@uses_fixture
def test_callable(polynomial: Polynomial, expected: Expected):
    assert polynomial(expected.x[0]) == pytest.approx(expected.y[0])
    utils.assert_allclose(polynomial(expected.x), expected.y)


@uses_fixture
def test_prime(polynomial: Polynomial, expected: Expected):
    if expected.prime is not None:
        assert polynomial.prime(expected.x[0]) == pytest.approx(expected.prime[0])
        utils.assert_allclose(
            polynomial.prime(expected.x), expected.prime  # type: ignore
        )


@uses_fixture
def test_zeros(polynomial: Polynomial, expected: Expected):
    if expected.zeros is not None:
        utils.assert_allclose(polynomial.zeros(), expected.zeros)  # type: ignore


@uses_fixture
def test_weights(polynomial: Polynomial, expected: Expected):
    if expected.weights is not None:
        utils.assert_allclose(polynomial.weights(), expected.weights)  # type: ignore


@uses_fixture
def test_abscissa(polynomial: Polynomial, expected: Expected):
    if expected.abscissa is not None:
        utils.assert_allclose(polynomial.abscissa(), expected.abscissa)  # type: ignore


@uses_fixture
def test_pickleable(polynomial: Polynomial, _):
    import pickle

    copy: Polynomial = pickle.loads(pickle.dumps(polynomial))
    assert copy is not polynomial
    assert copy.order == polynomial.order

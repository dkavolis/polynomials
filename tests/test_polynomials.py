from typing import Iterable, Tuple, Union

import pytest

import numpy as np
import utils

from polynomials import Chebyshev, Hermite, Laguerre, Legendre, LegendreStieltjes


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
        self.y = np.asarray(y)
        self.x = np.ones_like(self.y) * np.asarray(x)
        self.domain = domain
        self.prime = None if prime is None else np.asarray(prime)
        self.zeros = None if zeros is None else np.asarray(zeros)
        self.weights = None if weights is None else np.asarray(weights)
        self.abscissa = None if abscissa is None else np.asarray(abscissa)


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
def expectation(request):
    return request.param(EXPECTED[request.param].order), EXPECTED[request.param]


def uses_fixture(f):
    def wrapper(expectation):
        f(*expectation)

    return wrapper


@uses_fixture
def test_properties(polynomial, expected):
    assert polynomial.order == expected.order

    copy = type(polynomial)(polynomial)
    assert copy is not polynomial
    assert copy.order == polynomial.order
    copy.order = 1
    assert copy.order == 1

    utils.assert_allclose(type(polynomial).domain, expected.domain)


@uses_fixture
def test_callable(polynomial, expected):
    assert polynomial(expected.x[0]) == pytest.approx(expected.y[0])
    utils.assert_allclose(polynomial(expected.x), expected.y)


@uses_fixture
def test_prime(polynomial, expected):
    if expected.prime is not None:
        assert polynomial.prime(expected.x[0]) == pytest.approx(expected.prime[0])
        utils.assert_allclose(polynomial.prime(expected.x), expected.prime)


@uses_fixture
def test_zeros(polynomial, expected):
    if expected.zeros is not None:
        utils.assert_allclose(polynomial.zeros(), expected.zeros)


@uses_fixture
def test_weights(polynomial, expected):
    if expected.weights is not None:
        utils.assert_allclose(polynomial.weights(), expected.weights)


@uses_fixture
def test_abscissa(polynomial, expected):
    if expected.abscissa is not None:
        utils.assert_allclose(polynomial.abscissa(), expected.abscissa)


@uses_fixture
def test_pickleable(polynomial, _):
    import pickle

    copy = pickle.loads(pickle.dumps(polynomial))
    assert copy is not polynomial
    assert copy.order == polynomial.order

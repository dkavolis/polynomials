#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Union
import numpy.typing as npt
import numpy as np

from polynomials.polynomials_cpp import (
    Legendre,
    Laguerre,
    Chebyshev,
    Hermite,
    LegendreStieltjes,
    LegendreSeries,
    LaguerreSeries,
    ChebyshevSeries,
    HermiteSeries,
    LegendreStieltjesSeries,
    LegendreProduct,
    LaguerreProduct,
    ChebyshevProduct,
    HermiteProduct,
    LegendreStieltjesProduct,
    LegendreProductSet,
    LaguerreProductSet,
    ChebyshevProductSet,
    HermiteProductSet,
    LegendreStieltjesProductSet,
    LegendreSequence,
    LaguerreSequence,
    ChebyshevSequence,
    HermiteSequence,
    LegendreStieltjesSequence,
    views,
)

Polynomial = Union[Laguerre, Legendre, Chebyshev, Hermite, LegendreStieltjes]

PolynomialSeries = Union[
    LaguerreSeries,
    LegendreSeries,
    ChebyshevSeries,
    HermiteSeries,
    LegendreStieltjesSeries,
]

PolynomialProduct = Union[
    LaguerreProduct,
    LegendreProduct,
    ChebyshevProduct,
    HermiteProduct,
    LegendreStieltjesProduct,
]

PolynomialProductSet = Union[
    LaguerreProductSet,
    LegendreProductSet,
    ChebyshevProductSet,
    HermiteProductSet,
    LegendreStieltjesProductSet,
]

PolynomialSequence = Union[
    LaguerreSequence,
    LegendreSequence,
    ChebyshevSequence,
    HermiteSequence,
    LegendreStieltjesSequence,
]

PolynomialProductSetView = Union[
    views.LaguerreProductSetView,
    views.LegendreProductSetView,
    views.ChebyshevProductSetView,
    views.HermiteProductSetView,
    views.LegendreStieltjesProductSetView,
]

PolynomialProductView = Union[
    views.LaguerreProductView,
    views.LegendreProductView,
    views.ChebyshevProductView,
    views.HermiteProductView,
    views.LegendreStieltjesProductView,
]

PolynomialView = Union[
    views.LaguerreView,
    views.LegendreView,
    views.ChebyshevView,
    views.HermiteView,
    views.LegendreStieltjesView,
]

# shapes are not yet supported so skip them...
ArrayF16 = npt.NDArray[np.float16]
ArrayF32 = npt.NDArray[np.float32]
ArrayF64 = npt.NDArray[np.float64]
ArrayFloat = Union[ArrayF16, ArrayF32, ArrayF64]

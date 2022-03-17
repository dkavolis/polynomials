#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from polynomials.polynomials_cpp import (
    views,
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
)

from polynomials.hints import (
    Polynomial,
    PolynomialProduct,
    PolynomialProductSet,
    PolynomialSequence,
    PolynomialSeries,
    PolynomialView,
    PolynomialProductView,
    PolynomialProductSetView,
    ArrayF32,
    ArrayF64,
    ArrayFloat,
)

# useless flake8
from .pce import (  # noqa: F401
    LegendrePCE,
    LaguerrePCE,
    ChebyshevPCE,
    HermitePCE,
    LegendreStieltjesPCE,
    fit_improvement,
    AdaptivePCE,
)


def get_include_dir() -> str:
    import pathlib

    return str(pathlib.Path(__file__).parent / "include")


__all__ = [
    "LegendrePCE",
    "LaguerrePCE",
    "ChebyshevPCE",
    "HermitePCE",
    "LegendreStieltjesPCE",
    "fit_improvement",
    "get_include_dir",
    "AdaptivePCE",
    "views",
    "Legendre",
    "Laguerre",
    "Chebyshev",
    "Hermite",
    "LegendreStieltjes",
    "LegendreSeries",
    "LaguerreSeries",
    "ChebyshevSeries",
    "HermiteSeries",
    "LegendreStieltjesSeries",
    "LegendreProduct",
    "LaguerreProduct",
    "ChebyshevProduct",
    "HermiteProduct",
    "LegendreStieltjesProduct",
    "LegendreProductSet",
    "LaguerreProductSet",
    "ChebyshevProductSet",
    "HermiteProductSet",
    "LegendreStieltjesProductSet",
    "LegendreSequence",
    "LaguerreSequence",
    "ChebyshevSequence",
    "HermiteSequence",
    "LegendreStieltjesSequence",
    "Polynomial",
    "PolynomialProduct",
    "PolynomialProductSet",
    "PolynomialSequence",
    "PolynomialSeries",
    "PolynomialView",
    "PolynomialProductView",
    "PolynomialProductSetView",
    "ArrayF32",
    "ArrayF64",
    "ArrayFloat",
]

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

from .polynomials_cpp import *  # noqa: F403 F401

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


def get_include_dir():
    import os

    return os.path.join(os.path.abspath(__file__), "include")


def all_cpp():
    from . import polynomials_cpp as cpp

    return [i for i in dir(cpp) if not i.startswith("_")]


__all__ = [
    "LegendrePCE",
    "LaguerrePCE",
    "ChebyshevPCE",
    "HermitePCE",
    "LegendreStieltjesPCE",
    "fit_improvement",
    "get_include_dir",
    "AdaptivePCE",
] + all_cpp()

del all_cpp

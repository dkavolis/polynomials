from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Type, Union, cast

import numpy as np

from .polynomials_cpp import (
    ChebyshevProductSet,
    HermiteProductSet,
    LaguerreProductSet,
    LegendreProductSet,
    LegendreStieltjesProductSet,
)

if TYPE_CHECKING:
    from .polynomials_cpp import Sobol, Real


class PCEMeta(type):
    def __new__(
        mcs, name, bases, namespace, is_abstract: bool = False, tensor_type: Type = None
    ):
        cls = super().__new__(mcs, name, bases, namespace)
        if not is_abstract:
            if not isinstance(tensor_type, type):
                raise ValueError(f"tensor_type must a type, got {tensor_type}")

            cls._TENSOR_TYPE = tensor_type

        return cls

    def make_pce(self, size: int, dimensions: int):
        return self._TENSOR_TYPE(size=size, dimensions=dimensions)

    def make_full_set(self, orders: Tuple[int]):
        return self._TENSOR_TYPE.full_set(orders)


class PCEBase(metaclass=PCEMeta, is_abstract=True):
    def __init__(
        self,
        linear_model,
        dimensions: int = 0,
        components: int = 0,
        full_set: Tuple[int] = None,
    ):
        self.linear_model = linear_model

        if full_set is None:
            self._pce = type(self).make_pce(size=components, dimensions=dimensions)
        else:
            self._pce = type(self).make_full_set(full_set)
        self._sobol: Sobol = None

    def __iter__(self):
        return iter(self._pce)

    @property
    def dimensions(self) -> int:
        return self._pce.dimensions

    @dimensions.setter
    def dimensions(self, value: int) -> None:
        self._pce.dimensions = value
        self._sobol = None

    @property
    def components(self) -> int:
        return len(self._pce)

    @components.setter
    def components(self, value: int) -> None:
        self._pce.resize(value)
        self._sobol = None

    @property
    def sobol(self) -> Sobol:
        if self._sobol is None:
            self._sobol = self._pce.sobol()
        return self._sobol

    def fit(self, x, y, X=None) -> np.ndarray:
        """
        Fit linear model to PCE

        Parameters
        ----------
        x : array-like of shape (n_samples, dimensions)
            samples
        y : array-like of shape (n_samples) or (n_samples, n_targets)
            sampled values
        X : optional array-like of shape (n_samples, components)
            can be optionally provided to save on recalculating if samples and
            components have not changed

        Returns
        -------
        X : np.ndarray of shape (n_samples, components)
        """
        x = np.asarray(x)
        if x.ndim > 2 or x.ndim == 0:
            raise ValueError("Only 1D/2D sample vectors are allowed")
        if x.ndim == 1:
            x = x[:, np.newaxis]

        n_samples = x.shape[0]
        y = np.asarray(y)
        if y.ndim > 2 or y.ndim == 0:
            raise ValueError("Only 1D/2D input vectors are allowed")
        if len(y) != n_samples:
            raise ValueError("Number of samples and inputs do not match")

        if X is None:
            X = np.ndarray(shape=[n_samples, self.components], dtype=np.double)
            for i, component in enumerate(self._pce):
                X[:, i] = component(x)
        else:
            X = np.asarray(X)

        self.linear_model.fit(X, y)
        coefficients = np.asarray(self.linear_model.coef_)
        intercept = self.linear_model.intercept_
        if coefficients.ndim == 2:
            coefficients = coefficients[0, :]
            intercept = intercept[0]
        if not self.linear_model.fit_intercept:
            intercept = None

        self._set_coefficients(self._pce, coefficients, intercept)
        self._sobol = None

        return X

    def target_pce(self, index: int):
        coefficients = np.asarray(self.linear_model.coef_)[index, :]
        intercept = None
        if self.linear_model.fit_intercept:
            intercept = self.linear_model.intercept_

        copy = type(self)(linear_model=self.linear_model, dimensions=0, components=0)
        # use copy constructor
        copy._pce = type(self._pce).__init__(self._pce)
        self._set_coefficients(copy._pce, coefficients, intercept)

        return copy

    def predict(self, x):
        return self._pce(x)

    def sensitivity(self, indices: Union[int, Tuple[int]]) -> Real:
        return self.sobol.sensitivity(indices)

    def total_sensitivity(self, index: int) -> Real:
        return self.sobol.total_sensitivity(index)

    def total_sensitivities(self) -> np.ndarray:
        sensitivities = np.zeros(self.dimensions, dtype=np.double)
        for i in range(self.dimensions):
            sensitivities[i] = self.total_sensitivity(i)

        return sensitivities

    @property
    def coefficients(self) -> np.ndarray:
        return self.linear_model.coef_

    @staticmethod
    def _set_coefficients(pce, coefficients, intercept=None):
        for i, component in enumerate(pce):
            component.coefficient = coefficients[i]

        if intercept is not None:
            try:
                index = pce.index([0] * pce.dimensions)
                pce[index].coefficient = intercept
            except IndexError:
                pass


class LegendrePCE(PCEBase, tensor_type=LegendreProductSet):
    pass


class LaguerrePCE(PCEBase, tensor_type=LaguerreProductSet):
    pass


class HermitePCE(PCEBase, tensor_type=HermiteProductSet):
    pass


class ChebyshevPCE(PCEBase, tensor_type=ChebyshevProductSet):
    pass


class LegendreStieltjesPCE(PCEBase, tensor_type=LegendreStieltjesProductSet):
    pass


def fit_improvement(
    sensitivities: np.ndarray, previous_sensitivities: np.ndarray
) -> float:
    sensitivities = np.asarray(sensitivities)
    previous_sensitivities = np.asarray(previous_sensitivities)
    differences = np.abs(sensitivities - previous_sensitivities)
    return cast(float, np.mean(differences))

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence, Type, Union

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

    def make_full_set(self, orders: Sequence[int]):
        return self._TENSOR_TYPE.full_set(orders)


class PCEBase(metaclass=PCEMeta, is_abstract=True):
    def __init__(
        self,
        linear_model,
        dimensions: int = 0,
        components: int = 0,
        full_set: Sequence[int] = None,
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

    def target_coefficients(self, target: int) -> np.ndarray:
        coefficients = np.array(self.linear_model.coef_[target, :], copy=True)

        if self.linear_model.fit_intercept:
            intercept = self.linear_model.intercept_[target]
            if intercept != 0:
                try:
                    index = self._pce.index([0] * self.dimensions)
                    coefficients[index] = intercept
                except IndexError:
                    pass

        return coefficients

    def set_target_coefficients(self, target: int, coefficients: np.ndarray) -> None:
        self.linear_model.coef_[target, :] = coefficients
        self._fixup_intercept(target)

    def add_target_coefficients(self, target: int, delta: np.ndarray) -> None:
        self.linear_model.coef_[target, :] += delta
        self._fixup_intercept(target)

    def _fixup_intercept(self, target: int) -> None:
        if self.linear_model.fit_intercept:
            try:
                index = self._pce.index([0] * self.dimensions)
                self.linear_model.intercept_[target] = self.linear_model.coef_[
                    target, index
                ]
                self.linear_model.coef_[target, index] = 0
            except IndexError:
                pass

    def predict(
        self, x: np.ndarray, targets: Union[int, Sequence[int]] = 0
    ) -> np.ndarray:
        def f(*args, **kwargs):
            return self._pce(*args, **kwargs)

        return self._eval_targets(targets, f, x)

    def sensitivity(
        self, indices: Union[int, Sequence[int]], targets: Union[int, Sequence[int]] = 0
    ) -> Union[Real, np.ndarray]:
        def f(*args, **kwargs):
            return self._pce.sobol().sensitivity(*args, **kwargs)

        return self._eval_targets(targets, f, indices)

    def total_sensitivity(
        self, indices: Union[int, Sequence[int]], targets: Union[int, Sequence[int]] = 0
    ) -> Union[Real, np.ndarray]:
        def f(*args, **kwargs):
            return self._pce.sobol().total_sensitivity(*args, **kwargs)

        return self._eval_targets(targets, f, indices)

    def total_sensitivities(self, targets: Union[int, Sequence[int]] = 0) -> np.ndarray:
        return self.total_sensitivity(np.arange(self.dimensions), targets)

    def _eval_targets(
        self, targets: Union[int, Sequence[int]], function: Callable, *args, **kwargs
    ):

        if isinstance(targets, int):
            if targets < 0:
                if self.linear_model.coef_.ndim == 2:
                    n_targets = self.linear_model.coef_.shape[0]
                else:
                    n_targets = 1
                targets = list(range(n_targets))
            else:
                targets = [targets]
        else:
            targets = list(targets)

        if n_targets == 1 or (len(targets) == 1 and targets[0] == 0):
            return function(*args, **kwargs)

        y = []
        try:
            for i in targets:
                self._pce.assign_coefficients(self.target_coefficients(i))
                y.append(function(*args, **kwargs))
        finally:
            self._pce.assign_coefficients(self.target_coefficients(0))

        if len(targets) == 1:
            return y[0]
        return np.asarray(y)

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
    sensitivities: np.ndarray,
    previous_sensitivities: np.ndarray,
    axis: int = None,
    out: np.ndarray = None,
) -> float:
    sensitivities = np.asarray(sensitivities)
    previous_sensitivities = np.asarray(previous_sensitivities)
    differences = np.abs(sensitivities - previous_sensitivities)
    return np.mean(differences, axis=axis, out=out)


class AdaptivePCE:
    def __init__(self, pce: PCEBase, tolerance: float, targets: int = 1):
        self.pce = pce
        self._sensitivities = np.zeros([pce.dimensions, targets], dtype=np.double)
        self._errors = np.ones([targets]) * np.inf
        self.tolerance = tolerance
        self._x: np.ndarray = np.zeros([0])
        self._y: np.ndarray = np.zeros([0])

    @property
    def sample_count(self) -> int:
        return len(self._x)

    @property
    def targets(self) -> int:
        return len(self._errors)

    @property
    def errors(self) -> np.ndarray:
        return self._errors

    @property
    def converged(self) -> bool:
        return np.all(self._errors <= self.tolerance)

    def fit(self, x, y) -> bool:
        y = np.asarray(y)
        targets = 1
        if y.ndim > 1:
            targets = y.shape[1]
        self._sensitivities = np.zeros([self.pce.dimensions, targets])
        return self.improve(x, y)

    def update_convergence(self) -> bool:
        sensitivities = self.pce.total_sensitivities(-1)
        self._errors = fit_improvement(sensitivities, self._sensitivities, axis=1)
        self._sensitivities = sensitivities
        return self.converged

    def improve(self, x, y, update_convergence: bool = True) -> bool:
        x = np.asarray(x)
        y = np.asarray(y)

        self.pce.fit(x, y)

        self._x = x
        self._y = y

        if update_convergence:
            return self.update_convergence()
        return self.converged

    def improve_extend(self, x, y, update_convergence: bool = True) -> bool:
        if self._x.size == 0:
            x = np.array(x, copy=True)
            y = np.array(y, copy=True)
        else:
            x = np.vstack((self._x, x))
            y = np.vstack((self._y, y))
        return self.improve(x, y, update_convergence)

    def predict(
        self, x: np.ndarray, targets: Union[int, Sequence[int]] = -1
    ) -> np.ndarray:
        return self.pce.predict(x, targets)

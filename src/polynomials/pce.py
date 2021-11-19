from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Sequence,
    Type,
    Union,
    Protocol,
    runtime_checkable,
    Optional,
)

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


@runtime_checkable
class LinearModel(Protocol):
    coef_: np.ndarray
    intercept_: Union[float, np.ndarray]
    fit_intercept: bool

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


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
        linear_model: LinearModel,
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

    def fit(self, x: np.ndarray, y: np.ndarray, X: np.ndarray = None) -> np.ndarray:
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
            X = np.ndarray(
                shape=[n_samples, self.components], dtype=np.double, order="F"
            )
            for i, component in enumerate(self._pce):
                X[:, i] = component(x).flatten()
        else:
            X = np.asarray(X, order="F")

        try:
            self.linear_model.fit(X, y)
        except ValueError as e:
            raise ValueError(
                f"""Linear model fitting failed with
    x:
{x}
    y:
{y}"""
            ) from e
        self.sync_with_linear_model(0)

        return X

    def sync_with_linear_model(self, target: Optional[int] = None) -> None:
        coefficients = np.asanyarray(self.linear_model.coef_)

        if self.linear_model.fit_intercept:
            intercept = np.asanyarray(self.linear_model.intercept_)
        else:
            intercept = None

        if target is None:
            target = 0
        else:
            if coefficients.ndim == 2:
                if target >= len(coefficients):
                    raise ValueError(
                        f"target {target} is out of range for {len(coefficients)} targets linear model"
                    )
            elif target != 0 and target != -1:
                raise ValueError(
                    f"target {target} is out range for single target linear model"
                )

        if coefficients.ndim == 2:
            coefficients = coefficients[target, :]

            if self.linear_model.fit_intercept:
                intercept = intercept[target]

        self._set_coefficients(self._pce, coefficients, intercept)
        self._sobol = None

    def target_coefficients(
        self, target: Optional[int], copy: bool = True
    ) -> np.ndarray:
        if target is None:
            s = slice(None)
        else:
            s = target

        if self.linear_model.coef_.ndim == 2:
            coefficients = np.array(self.linear_model.coef_[s, :], copy=copy)
        elif target is None or target < 0:
            coefficients = np.array(self.linear_model.coef_, copy=copy)
        else:
            raise ValueError(f"Invalid target, only target 0 exists but got {target}")

        if not copy:
            return coefficients

        if self.linear_model.fit_intercept:
            try:
                intercept = self.linear_model.intercept_[s]
            except IndexError:
                intercept = self.linear_model.intercept_

            try:
                index = self._pce.index([0] * self.dimensions)
                if self.dimensions == 1:
                    index = index[0]
            except IndexError:
                pass
            else:
                if coefficients.ndim == 2:
                    coefficients[s, index] = intercept
                else:
                    coefficients[index] = intercept

        return coefficients

    def set_target_coefficients(self, target: int, coefficients: np.ndarray) -> None:
        self.target_coefficients(target, False)[:] = coefficients
        self._fixup_intercept(target, lambda _, x: x)

    def add_target_coefficients(self, target: int, delta: np.ndarray) -> None:
        self.target_coefficients(target, False)[:] += delta

        def add_intercept(existing: float, value: float) -> float:
            return existing + value

        self._fixup_intercept(target, add_intercept)

    def _fixup_intercept(
        self, target: int, update: Callable[[float, float], float]
    ) -> None:
        intercept = None
        if self.linear_model.fit_intercept:
            try:
                index = self._pce.index([0] * self.dimensions)
                if self.dimensions == 1:
                    index = index[0]
                old = self.linear_model.intercept_[target]
                new = self.target_coefficients(target, False)[index]
                intercept = update(old, new)
                self.linear_model.intercept_[target] = intercept
                self.target_coefficients(target, False)[index] = 0
            except IndexError:
                pass

        if target == 0:
            self._set_coefficients(
                self._pce, self.target_coefficients(target, False), intercept
            )

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

        if len(targets) == 1 and targets[0] == 0:
            # assume synced already since only 1 target
            return function(*args, **kwargs)

        y = []
        for i in targets:
            self.sync_with_linear_model(i)
            y.append(function(*args, **kwargs))

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
        if pce.dimensions == 1:
            self._sensitivities = np.zeros([targets], dtype=np.double)
        else:
            self._sensitivities = np.zeros([pce.dimensions, targets], dtype=np.double)
        self._errors: np.ndarray = np.ones([targets]) * np.inf
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
    def sensitivities(self) -> np.ndarray:
        return self._sensitivities

    @property
    def converged(self) -> bool:
        return np.all(self._errors <= self.tolerance)

    def fit(self, x: np.ndarray, y: np.ndarray) -> bool:
        y = np.asarray(y)
        targets = 1
        if y.ndim > 1:
            targets = y.shape[1]
        self._sensitivities = np.zeros([self.pce.dimensions, targets])
        return self.improve(x, y)

    def update_convergence(self) -> bool:
        if self.pce.dimensions == 1:
            # variance is the sum of non-zero term coefficients squared
            std = np.array(self.pce.linear_model.coef_)
            std = np.linalg.norm(std, axis=-1)

            # for consistency with multi-variate polynomials use relative change in
            # standard deviation as the error
            self._errors = np.abs(1.0 - std / self._sensitivities)
            self._sensitivities = std
        else:
            sensitivities = self.pce.total_sensitivities(-1)

            self._errors = fit_improvement(sensitivities, self._sensitivities, axis=-1)
            self._sensitivities = sensitivities
        return self.converged

    def improvement_value(
        self, old_sensitivies: np.ndarray
    ) -> Union[np.float64, np.ndarray]:
        if self.pce.dimensions == 1:
            return np.abs(self._sensitivities - old_sensitivies)
        return fit_improvement(self._sensitivities, old_sensitivies, axis=-1)

    def has_converged(self, old_sensitivies: np.ndarray) -> bool:
        errors = self.improvement_value(old_sensitivies)
        return np.all(errors <= self.tolerance)

    def improve(
        self, x: np.ndarray, y: np.ndarray, update_convergence: bool = True
    ) -> bool:
        x = np.asarray(x)
        y = np.asarray(y)

        self.pce.fit(x, y)

        self._x = x
        self._y = y

        if update_convergence:
            return self.update_convergence()
        return self.converged

    def improve_extend(
        self, x: np.ndarray, y: np.ndarray, update_convergence: bool = True
    ) -> bool:
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

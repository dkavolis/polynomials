#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    Protocol,
    cast,
    runtime_checkable,
    Optional,
)

import numpy as np
import numpy.typing as npt

from polynomials.polynomials_cpp import (
    ChebyshevProductSet,
    HermiteProductSet,
    LaguerreProductSet,
    LegendreProductSet,
    LegendreStieltjesProductSet,
)
from polynomials.hints import (
    Polynomial,
    PolynomialProductSet,
    PolynomialProductSetView,
    ArrayFloat,
)

if TYPE_CHECKING:
    from typing_extensions import ParamSpec
    from polynomials.polynomials_cpp import Sobol

    P = ParamSpec("P")

R = TypeVar("R")
N = TypeVar("N")


@runtime_checkable
class LinearModel(Protocol):
    coef_: ArrayFloat
    intercept_: Union[float, ArrayFloat]
    fit_intercept: bool

    def fit(
        self, X: ArrayFloat, y: ArrayFloat, sample_weight: Optional[ArrayFloat]
    ) -> None:
        ...

    def predict(self, X: ArrayFloat) -> ArrayFloat:
        ...

    def score(
        self, X: ArrayFloat, y: ArrayFloat, sample_weight: Optional[ArrayFloat] = ...
    ) -> float:
        ...


T = TypeVar("T")


class PCEMeta(type):
    TENSOR_TYPE: Type[PolynomialProductSet]

    def __new__(
        cls: Type[PCEMeta],
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        is_abstract: bool = False,
        tensor_type: Type[PolynomialProductSet] = None,
    ) -> PCEMeta:
        klass = super().__new__(cls, name, bases, namespace)
        if not is_abstract:
            if not isinstance(tensor_type, type):
                raise ValueError(f"tensor_type must a type, got {tensor_type}")

            setattr(klass, "TENSOR_TYPE", tensor_type)

        return klass

    def make_pce(self: PCEMeta, size: int, dimensions: int) -> PolynomialProductSet:
        return self.TENSOR_TYPE(size=size, dimensions=dimensions)

    def make_full_set(self, orders: Iterable[int]) -> PolynomialProductSet:
        return self.TENSOR_TYPE.full_set(orders)


PCEBaseType = TypeVar("PCEBaseType", bound="PCEBase")


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
        self._sample_array: Optional[ArrayFloat] = None

    @property
    def X_(self) -> ArrayFloat:
        if self._sample_array is None:
            raise ValueError("Tried to access sample_array before fitting")
        return self._sample_array

    @property
    def polynomial_type(self) -> Type[PolynomialProductSet]:
        return type(self).TENSOR_TYPE

    @property
    def index_set(self) -> Set[Tuple[int, ...]]:
        return set(
            tuple(cast(Polynomial, polynomial).order for polynomial in product)
            for product in self
        )

    def __iter__(self) -> Iterator[PolynomialProductSetView]:
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

    def fit(
        self: PCEBaseType,
        x: ArrayFloat,
        y: ArrayFloat,
        sample_weight: Optional[ArrayFloat] = None,
        X: Optional[ArrayFloat] = None,
    ) -> PCEBaseType:
        """
        Fit linear model to PCE

        Parameters
        ----------
        x : array-like of shape (n_samples, dimensions)
            samples
        y : array-like of shape (n_samples) or (n_samples, n_targets)
            sampled values
        sample_weight: optional array-like of shape (n_samples,)
            sample weights
        X : optional array-like of shape (n_samples, components)
            can be optionally provided to save on recalculating if samples and
            components have not changed

        Returns
        -------
        self
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
            X = self.sample_array(x)  # type: ignore # constant...
        else:
            X = np.asarray(X, order="F")  # type: ignore # constant...

        try:
            self.linear_model.fit(X, y, sample_weight=sample_weight)
        except ValueError as e:
            raise ValueError(
                f"""Linear model fitting failed with
    x:
{x}
    y:
{y}"""
            ) from e
        self.sync_with_linear_model(0)
        self._sample_array = X

        return self

    def sync_with_linear_model(self, target: Optional[int] = None) -> None:
        coefficients: ArrayFloat = np.asanyarray(self.linear_model.coef_)

        intercept: Optional[ArrayFloat]
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
                        f"target {target} is out of range for {len(coefficients)}"
                        " targets linear model"
                    )
            elif target != 0 and target != -1:
                raise ValueError(
                    f"target {target} is out range for single target linear model"
                )

        if coefficients.ndim == 2:
            coefficients = coefficients[target, :]

        # np.asanyarray always returns an array, if intercept was float it would be an
        # array of size 1 now
        if intercept is not None:
            intercept_ = intercept[target]
        else:
            intercept_ = None

        self._set_coefficients(self._pce, coefficients, intercept_)
        self._sobol = None

    def sample_array(self, x: ArrayFloat) -> ArrayFloat:
        x = np.asanyarray(x)

        X = np.empty(  # type: ignore # constant...
            shape=[len(x), self.components], dtype=np.double, order="F"
        )
        for i, component in enumerate(self):
            X[:, i] = component(x).flatten()

        return X

    def target_coefficients(
        self, target: Optional[int], copy: bool = True
    ) -> ArrayFloat:
        s: Union[slice, int]
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
                intercept = self.linear_model.intercept_[s]  # type: ignore # try/except
            except IndexError:
                intercept = self.linear_model.intercept_

            index = self._intercept_index(self._pce)
            if index is not None:
                if coefficients.ndim == 2:
                    coefficients[s, index] = intercept
                else:
                    coefficients[index] = intercept

        return coefficients

    def set_target_coefficients(self, target: int, coefficients: ArrayFloat) -> None:
        self.target_coefficients(target, False)[:] = coefficients
        self._fixup_intercept(target, lambda _, x: x)

    def add_target_coefficients(self, target: int, delta: ArrayFloat) -> None:
        self.target_coefficients(target, False)[:] += delta

        def add_intercept(existing: float, value: float) -> float:
            return existing + value

        self._fixup_intercept(target, add_intercept)

    @staticmethod
    def _intercept_index(pce: PolynomialProductSet) -> Optional[int]:
        try:
            i: Union[int, ArrayFloat] = pce.index([0] * pce.dimensions)
            index: int
            if isinstance(i, int):
                index = i
            else:
                index = i[0]
        except IndexError:
            return None
        else:
            return index

    def _fixup_intercept(
        self, target: int, update: Callable[[float, float], float]
    ) -> None:
        intercept = None
        if self.linear_model.fit_intercept:
            index = self._intercept_index(self._pce)
            if index is not None:
                try:
                    old: float = self.linear_model.intercept_[target]  # type: ignore
                except TypeError:
                    pass
                else:
                    new = self.target_coefficients(target, False)[index]
                    intercept = update(old, new)
                    self.linear_model.intercept_[target] = intercept  # type: ignore
                    self.target_coefficients(target, False)[index] = 0

        if target == 0:
            self._set_coefficients(
                self._pce, self.target_coefficients(target, False), intercept
            )

    def predict(
        self, x: ArrayFloat, targets: Union[int, Sequence[int]] = 0
    ) -> ArrayFloat:
        def f(xx: ArrayFloat) -> ArrayFloat:
            return self._pce(xx)

        return self._eval_targets(targets, f, x)

    def score(
        self,
        x: ArrayFloat,
        y: ArrayFloat,
        sample_weight: Optional[ArrayFloat] = None,
        X: Optional[ArrayFloat] = None,
    ) -> float:
        if X is None:
            X = self.sample_array(x)  # type: ignore
        return self.linear_model.score(X, y, sample_weight=sample_weight)

    def sensitivity(
        self, indices: Union[int, Sequence[int]], targets: Union[int, Sequence[int]] = 0
    ) -> Union[float, ArrayFloat]:
        def f(xx: ArrayFloat) -> Union[float, ArrayFloat]:
            return self._pce.sobol().sensitivity(xx)

        return self._eval_targets(targets, f, indices)

    def total_sensitivity(
        self, indices: Union[int, Iterable[int]], targets: Union[int, Iterable[int]] = 0
    ) -> Union[float, ArrayFloat]:
        def f(xx: ArrayFloat) -> Union[float, ArrayFloat]:
            return self._pce.sobol().total_sensitivity(xx)

        return self._eval_targets(targets, f, indices)

    def total_sensitivities(self, targets: Union[int, Iterable[int]] = 0) -> ArrayFloat:
        return np.asanyarray(
            self.total_sensitivity(
                cast(npt.NDArray[np.int32], np.arange(self.dimensions)),
                targets,
            )
        )

    # seems mypy still doesn't like ParamSpec but the usage is valid
    # https://www.python.org/dev/peps/pep-0612/#toc-entry-12
    def _eval_targets(
        self,
        targets: Union[int, Iterable[int]],
        function: Callable[P, R],  # type: ignore
        *args: P.args,  # type: ignore
        **kwargs: P.kwargs,  # type: ignore
    ) -> Union[R, ArrayFloat]:

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

        y: List[R] = []
        for i in targets:
            self.sync_with_linear_model(i)
            y.append(function(*args, **kwargs))

        if len(targets) == 1:
            return y[0]
        return np.asarray(y)

    @property
    def coefficients(self) -> ArrayFloat:
        return self.linear_model.coef_

    @staticmethod
    def _set_coefficients(
        pce: PolynomialProductSet,
        coefficients: ArrayFloat,
        intercept: Optional[float] = None,
    ):
        component: PolynomialProductSetView
        for i, component in enumerate(pce):
            component.coefficient = coefficients[i]

        if intercept is not None:
            index = PCEBase._intercept_index(pce)
            if index is not None:
                pce[index].coefficient = intercept


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
    sensitivities: ArrayFloat,
    previous_sensitivities: ArrayFloat,
    axis: int = None,
    out: ArrayFloat = None,
) -> Union[float, ArrayFloat]:
    sensitivities = np.asarray(sensitivities)
    previous_sensitivities = np.asarray(previous_sensitivities)
    differences = np.abs(sensitivities - previous_sensitivities)
    return np.mean(differences, axis=axis, out=out)


class AdaptivePCE:
    def __init__(self, pce: PCEBase, tolerance: float, targets: int = 1):
        self.pce = pce
        self._sensitivities: ArrayFloat
        if pce.dimensions == 1:
            self._sensitivities = np.zeros([targets], dtype=np.double)
        else:
            self._sensitivities = np.zeros([pce.dimensions, targets], dtype=np.double)
        self._errors: ArrayFloat = np.ones([targets]) * np.inf
        self.tolerance = tolerance
        self._x: ArrayFloat = np.zeros([0])
        self._y: ArrayFloat = np.zeros([0])

    @property
    def sample_count(self) -> int:
        return len(self._x)

    @property
    def targets(self) -> int:
        return len(self._errors)

    @property
    def errors(self) -> ArrayFloat:
        return self._errors

    @property
    def sensitivities(self) -> ArrayFloat:
        return self._sensitivities

    @property
    def converged(self) -> bool:
        return cast(bool, np.all(cast(ArrayFloat, self._errors <= self.tolerance)))

    def fit(
        self, x: ArrayFloat, y: ArrayFloat, sample_weight: Optional[ArrayFloat] = None
    ) -> bool:
        y = np.asarray(y)
        targets = 1
        if y.ndim > 1:
            targets = y.shape[1]
        self._sensitivities = np.zeros([self.pce.dimensions, targets])
        return self.improve(x, y, sample_weight=sample_weight)

    def update_convergence(self) -> bool:
        prev_sensitivities = self._sensitivities
        if self.pce.dimensions == 1:
            # variance is the sum of non-zero term coefficients squared
            self._sensitivities = np.linalg.norm(self.pce.linear_model.coef_, axis=-1)
        else:
            self._sensitivities = self.pce.total_sensitivities(-1)
        self._errors = np.asanyarray(self.improvement_value(prev_sensitivities))

        return self.converged

    def improvement_value(
        self, old_sensitivies: ArrayFloat
    ) -> Union[float, ArrayFloat]:
        if self.pce.dimensions == 1:
            # for consistency with multi-variate polynomials use relative change in
            # standard deviation as the error
            return np.abs(1.0 - self._sensitivities / old_sensitivies)
        return fit_improvement(self._sensitivities, old_sensitivies, axis=-1)

    def has_converged(self, old_sensitivies: ArrayFloat) -> bool:
        errors = self.improvement_value(old_sensitivies)
        return cast(bool, cast(ArrayFloat, errors <= self.tolerance).all())

    def improve(
        self,
        x: ArrayFloat,
        y: ArrayFloat,
        update_convergence: bool = True,
        sample_weight: Optional[ArrayFloat] = None,
        X: Optional[ArrayFloat] = None,
    ) -> bool:
        x = np.asarray(x)
        y = np.asarray(y)

        self.pce.fit(x, y, sample_weight=sample_weight, X=X)

        self._x = x
        self._y = y

        if update_convergence:
            return self.update_convergence()
        return self.converged

    def improve_extend(
        self,
        x: ArrayFloat,
        y: ArrayFloat,
        update_convergence: bool = True,
        X: Optional[ArrayFloat] = None,
    ) -> bool:
        if self._x.size == 0:
            x = np.array(x, copy=True)
            y = np.array(y, copy=True)
        else:
            x = np.vstack((self._x, x))
            y = np.vstack((self._y, y))
        return self.improve(x, y, update_convergence, X=X)

    def predict(
        self, x: ArrayFloat, targets: Union[int, Sequence[int]] = -1
    ) -> ArrayFloat:
        return self.pce.predict(x, targets)

    def score(
        self,
        x: ArrayFloat,
        y: ArrayFloat,
        sample_weight: Optional[ArrayFloat] = None,
        X: Optional[ArrayFloat] = None,
    ) -> float:
        return self.pce.score(x, y, sample_weight=sample_weight, X=X)

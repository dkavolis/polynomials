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
    Tuple,
    Type,
    TypeVar,
    Union,
    Protocol,
    cast,
    runtime_checkable,
    Optional,
)
from typing_extensions import ParamSpec

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
    PolynomialProductSet,
    PolynomialProductSetView,
)

if TYPE_CHECKING:
    from polynomials.polynomials_cpp import Sobol

# shapes are not yet supported so skip them...
FloatArray = Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]

P = ParamSpec("P")
R = TypeVar("R")
N = TypeVar("N")


@runtime_checkable
class LinearModel(Protocol):
    coef_: FloatArray
    intercept_: Union[float, FloatArray]
    fit_intercept: bool

    def fit(self, X: FloatArray, y: FloatArray) -> None:
        ...

    def predict(self, X: FloatArray) -> FloatArray:
        ...

    def score(
        self, X: FloatArray, y: FloatArray, sample_weight: Optional[FloatArray] = ...
    ) -> float:
        ...


T = TypeVar("T")


class PCEMeta(type):
    _TENSOR_TYPE: Type[PolynomialProductSet]

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

            setattr(klass, "_TENSOR_TYPE", tensor_type)

        return klass

    def make_pce(self: PCEMeta, size: int, dimensions: int) -> PolynomialProductSet:
        return self._TENSOR_TYPE(size=size, dimensions=dimensions)

    def make_full_set(self, orders: Iterable[int]) -> PolynomialProductSet:
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

    def fit(self, x: FloatArray, y: FloatArray, X: FloatArray = None) -> FloatArray:
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
            X = self._sample_array(x)  # type: ignore # constant...
        else:
            X = np.asarray(X, order="F")  # type: ignore # constant...

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
        coefficients: FloatArray = np.asanyarray(self.linear_model.coef_)

        intercept: Optional[FloatArray]
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

    def _sample_array(self, x: FloatArray) -> FloatArray:
        x = np.asanyarray(x)

        X = np.empty(  # type: ignore # constant...
            shape=[len(x), self.components], dtype=np.double, order="F"
        )
        for i, component in enumerate(self):
            X[:, i] = component(x).flatten()

        return X

    def target_coefficients(
        self, target: Optional[int], copy: bool = True
    ) -> FloatArray:
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

    def set_target_coefficients(self, target: int, coefficients: FloatArray) -> None:
        self.target_coefficients(target, False)[:] = coefficients
        self._fixup_intercept(target, lambda _, x: x)

    def add_target_coefficients(self, target: int, delta: FloatArray) -> None:
        self.target_coefficients(target, False)[:] += delta

        def add_intercept(existing: float, value: float) -> float:
            return existing + value

        self._fixup_intercept(target, add_intercept)

    @staticmethod
    def _intercept_index(pce: PolynomialProductSet) -> Optional[int]:
        try:
            i: Union[int, FloatArray] = pce.index([0] * pce.dimensions)
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
        self, x: FloatArray, targets: Union[int, Sequence[int]] = 0
    ) -> FloatArray:
        def f(xx: FloatArray) -> FloatArray:
            return self._pce(xx)

        return self._eval_targets(targets, f, x)

    def score(
        self, x: FloatArray, y: FloatArray, sample_weight: Optional[FloatArray] = None
    ) -> float:
        X = self._sample_array(x)
        return self.linear_model.score(X, y, sample_weight=sample_weight)

    def sensitivity(
        self, indices: Union[int, Sequence[int]], targets: Union[int, Sequence[int]] = 0
    ) -> Union[float, FloatArray]:
        def f(xx: FloatArray) -> Union[float, FloatArray]:
            return self._pce.sobol().sensitivity(xx)

        return self._eval_targets(targets, f, indices)

    def total_sensitivity(
        self, indices: Union[int, Iterable[int]], targets: Union[int, Iterable[int]] = 0
    ) -> Union[float, FloatArray]:
        def f(xx: FloatArray) -> Union[float, FloatArray]:
            return self._pce.sobol().total_sensitivity(xx)

        return self._eval_targets(targets, f, indices)

    def total_sensitivities(self, targets: Union[int, Iterable[int]] = 0) -> FloatArray:
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
    ) -> Union[R, FloatArray]:

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
    def coefficients(self) -> FloatArray:
        return self.linear_model.coef_

    @staticmethod
    def _set_coefficients(
        pce: PolynomialProductSet,
        coefficients: FloatArray,
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
    sensitivities: FloatArray,
    previous_sensitivities: FloatArray,
    axis: int = None,
    out: FloatArray = None,
) -> Union[float, FloatArray]:
    sensitivities = np.asarray(sensitivities)
    previous_sensitivities = np.asarray(previous_sensitivities)
    differences = np.abs(sensitivities - previous_sensitivities)
    return np.mean(differences, axis=axis, out=out)


class AdaptivePCE:
    def __init__(self, pce: PCEBase, tolerance: float, targets: int = 1):
        self.pce = pce
        self._sensitivities: FloatArray
        if pce.dimensions == 1:
            self._sensitivities = np.zeros([targets], dtype=np.double)
        else:
            self._sensitivities = np.zeros([pce.dimensions, targets], dtype=np.double)
        self._errors: FloatArray = np.ones([targets]) * np.inf
        self.tolerance = tolerance
        self._x: FloatArray = np.zeros([0])
        self._y: FloatArray = np.zeros([0])

    @property
    def sample_count(self) -> int:
        return len(self._x)

    @property
    def targets(self) -> int:
        return len(self._errors)

    @property
    def errors(self) -> FloatArray:
        return self._errors

    @property
    def sensitivities(self) -> FloatArray:
        return self._sensitivities

    @property
    def converged(self) -> bool:
        return cast(bool, np.all(cast(FloatArray, self._errors <= self.tolerance)))

    def fit(self, x: FloatArray, y: FloatArray) -> bool:
        y = np.asarray(y)
        targets = 1
        if y.ndim > 1:
            targets = y.shape[1]
        self._sensitivities = np.zeros([self.pce.dimensions, targets])
        return self.improve(x, y)

    def update_convergence(self) -> bool:
        if self.pce.dimensions == 1:
            # variance is the sum of non-zero term coefficients squared
            std: FloatArray = np.array(self.pce.linear_model.coef_)
            std = np.linalg.norm(std, axis=-1)

            # for consistency with multi-variate polynomials use relative change in
            # standard deviation as the error
            self._errors = np.abs(1.0 - std / self._sensitivities)
            self._sensitivities = std
        else:
            sensitivities = self.pce.total_sensitivities(-1)

            self._errors = np.asanyarray(
                fit_improvement(sensitivities, self._sensitivities, axis=-1)
            )
            self._sensitivities = sensitivities
        return self.converged

    def improvement_value(
        self, old_sensitivies: FloatArray
    ) -> Union[float, FloatArray]:
        if self.pce.dimensions == 1:
            return np.abs(self._sensitivities - old_sensitivies)
        return fit_improvement(self._sensitivities, old_sensitivies, axis=-1)

    def has_converged(self, old_sensitivies: FloatArray) -> bool:
        errors = self.improvement_value(old_sensitivies)
        return cast(bool, cast(FloatArray, errors <= self.tolerance).all())

    def improve(
        self, x: FloatArray, y: FloatArray, update_convergence: bool = True
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
        self, x: FloatArray, y: FloatArray, update_convergence: bool = True
    ) -> bool:
        if self._x.size == 0:
            x = np.array(x, copy=True)
            y = np.array(y, copy=True)
        else:
            x = np.vstack((self._x, x))
            y = np.vstack((self._y, y))
        return self.improve(x, y, update_convergence)

    def predict(
        self, x: FloatArray, targets: Union[int, Sequence[int]] = -1
    ) -> FloatArray:
        return self.pce.predict(x, targets)

    def score(
        self, x: FloatArray, y: FloatArray, sample_weight: Optional[FloatArray] = None
    ) -> float:
        return self.pce.score(x, y, sample_weight)

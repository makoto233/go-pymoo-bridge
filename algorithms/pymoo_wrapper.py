from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.result import Result

from algorithms.base import AlgorithmConfig, AlgorithmStepResult, OptimizationAlgorithm


class DummyProblem(Problem):
    """Placeholder problem used only to satisfy Pymoo's initialization contract."""

    def __init__(
        self,
        *,
        n_var: int,
        n_obj: int,
        n_ieq_constr: int,
        xl: Sequence[float],
        xu: Sequence[float],
    ) -> None:
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_ieq_constr=n_ieq_constr,
            xl=np.asarray(xl, dtype=float),
            xu=np.asarray(xu, dtype=float),
        )

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "DummyProblem does not evaluate fitness. Use the ask-and-tell API instead."
        )


class PymooWrapper(OptimizationAlgorithm):
    """Generic adapter that exposes Pymoo algorithms through a shared interface."""

    def __init__(self, config: AlgorithmConfig, algorithm: Algorithm) -> None:
        super().__init__(config=config)
        self.algorithm = algorithm
        self.problem = DummyProblem(
            n_var=config.n_var,
            n_obj=config.n_obj,
            n_ieq_constr=config.n_ieq_constr,
            xl=config.xl,
            xu=config.xu,
        )
        self._initialized = False

    def init(self) -> list[list[float]]:
        if self._initialized:
            raise RuntimeError("Optimizer has already been initialized.")

        self.algorithm.setup(
            self.problem,
            termination=("n_gen", self.config.n_gen),
            seed=self.config.seed,
            verbose=self.config.verbose,
        )
        self._initialized = True

        initial_population = self.algorithm.ask()
        return self._matrix_to_list(initial_population.get("X"), expected_width=self.config.n_var)

    def step(
        self,
        x: Sequence[Sequence[float]],
        f: Sequence[Sequence[float]] | Sequence[float],
        g: Sequence[Sequence[float]] | Sequence[float] | None = None,
    ) -> AlgorithmStepResult:
        if not self._initialized:
            raise RuntimeError("Optimizer has not been initialized.")

        x_array = self._reshape_matrix(x, expected_width=self.config.n_var, field_name="X")
        f_array = self._reshape_matrix(f, expected_width=self.config.n_obj, field_name="F")

        if len(x_array) != len(f_array):
            raise ValueError("X and F must contain the same number of rows.")

        population_kwargs: dict[str, np.ndarray] = {"X": x_array, "F": f_array}
        g_array: np.ndarray | None = None

        if self.config.n_ieq_constr > 0:
            if g is None:
                raise ValueError("This task expects G because constraints were configured.")
            g_array = self._reshape_matrix(
                g,
                expected_width=self.config.n_ieq_constr,
                field_name="G",
            )
            if len(g_array) != len(x_array):
                raise ValueError("G and X must contain the same number of rows.")
            population_kwargs["G"] = g_array
        elif g is not None:
            raise ValueError("G was provided, but this task was created without constraints.")

        population = Population.new(*self._flatten_population_kwargs(population_kwargs))
        tell_result = self.algorithm.tell(population)

        best_x, best_f, best_g = self._extract_best_payload()

        if isinstance(tell_result, Result) or not self.algorithm.has_next():
            final_result = self.algorithm.result()
            return AlgorithmStepResult(
                done=True,
                generation=int(self.algorithm.n_gen),
                next_x=None,
                best_x=self._matrix_to_list(final_result.X, expected_width=self.config.n_var),
                best_f=self._matrix_to_list(final_result.F, expected_width=self.config.n_obj),
                best_g=self._matrix_to_list(
                    getattr(final_result, "G", None),
                    expected_width=self.config.n_ieq_constr,
                ),
            )

        next_population = self.algorithm.ask()
        next_x = self._matrix_to_list(next_population.get("X"), expected_width=self.config.n_var)
        return AlgorithmStepResult(
            done=False,
            generation=int(self.algorithm.n_gen),
            next_x=next_x,
            best_x=best_x,
            best_f=best_f,
            best_g=best_g,
        )

    def _extract_best_payload(
        self,
    ) -> tuple[list[list[float]] | None, list[list[float]] | None, list[list[float]] | None]:
        opt_population = getattr(self.algorithm, "opt", None)
        if opt_population is None or len(opt_population) == 0:
            return None, None, None

        best_x = self._matrix_to_list(opt_population.get("X"), expected_width=self.config.n_var)
        best_f = self._matrix_to_list(opt_population.get("F"), expected_width=self.config.n_obj)
        best_g = self._matrix_to_list(
            opt_population.get("G"),
            expected_width=self.config.n_ieq_constr,
        )
        return best_x, best_f, best_g

    @staticmethod
    def _flatten_population_kwargs(population_kwargs: dict[str, np.ndarray]) -> list[Any]:
        flattened: list[Any] = []
        for key, value in population_kwargs.items():
            flattened.extend([key, value])
        return flattened

    @staticmethod
    def _reshape_matrix(
        values: Sequence[Sequence[float]] | Sequence[float],
        *,
        expected_width: int,
        field_name: str,
    ) -> np.ndarray:
        if expected_width <= 0:
            raise ValueError(f"{field_name} cannot be reshaped because its width is zero.")

        array = np.asarray(values, dtype=float)
        if array.size == 0:
            raise ValueError(f"{field_name} cannot be empty.")

        try:
            return array.reshape(-1, expected_width)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} could not be reshaped to (-1, {expected_width})."
            ) from exc

    @staticmethod
    def _matrix_to_list(
        values: np.ndarray | Sequence[float] | Sequence[Sequence[float]] | None,
        *,
        expected_width: int,
    ) -> list[list[float]] | None:
        if values is None or expected_width <= 0:
            return None

        array = np.asarray(values, dtype=float)
        if array.size == 0:
            return []

        try:
            return array.reshape(-1, expected_width).tolist()
        except ValueError as exc:
            raise ValueError(
                f"Result payload could not be reshaped to (-1, {expected_width})."
            ) from exc

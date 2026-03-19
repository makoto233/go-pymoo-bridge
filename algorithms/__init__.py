from __future__ import annotations

from math import comb
from functools import partial
from typing import Callable

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.util.ref_dirs import get_reference_directions

from algorithms.base import AlgorithmConfig, OptimizationAlgorithm
from algorithms.pymoo_wrapper import PymooWrapper

AlgorithmBuilder = Callable[[AlgorithmConfig], OptimizationAlgorithm]


def _build_pymoo_wrapper(
    config: AlgorithmConfig,
    *,
    algorithm_cls: type,
    default_params: dict | None = None,
) -> OptimizationAlgorithm:
    params = dict(default_params or {})
    params.update(dict(config.algorithm_params))
    params.setdefault("pop_size", config.pop_size)
    return PymooWrapper(config=config, algorithm=algorithm_cls(**params))


def _resolve_reference_directions(config: AlgorithmConfig, params: dict) -> object:
    ref_dirs = params.pop("ref_dirs", None)
    if ref_dirs is not None:
        return ref_dirs

    n_partitions = params.pop("n_partitions", None)
    if n_partitions is None:
        n_partitions = _choose_reference_partitions(config.n_obj, config.pop_size)

    return get_reference_directions("das-dennis", config.n_obj, n_partitions=int(n_partitions))


def _choose_reference_partitions(n_obj: int, pop_size: int) -> int:
    if n_obj <= 1:
        raise ValueError("NSGA3 requires n_obj greater than 1.")

    for partitions in range(1, max(pop_size, 2) + 1):
        if comb(n_obj + partitions - 1, partitions) >= pop_size:
            return partitions

    return max(pop_size - 1, 1)


def _build_nsga3_wrapper(config: AlgorithmConfig) -> OptimizationAlgorithm:
    params = dict(config.algorithm_params)
    params.setdefault("eliminate_duplicates", True)
    params.setdefault("pop_size", config.pop_size)
    params["ref_dirs"] = _resolve_reference_directions(config, params)
    return PymooWrapper(config=config, algorithm=NSGA3(**params))


ALGORITHM_REGISTRY: dict[str, AlgorithmBuilder] = {
    "ga": partial(_build_pymoo_wrapper, algorithm_cls=GA),
    "nsga2": partial(
        _build_pymoo_wrapper,
        algorithm_cls=NSGA2,
        default_params={"eliminate_duplicates": True},
    ),
    "nsga3": _build_nsga3_wrapper,
    "pso": partial(_build_pymoo_wrapper, algorithm_cls=PSO),
}


def create_algorithm(name: str, config: AlgorithmConfig) -> OptimizationAlgorithm:
    builder = ALGORITHM_REGISTRY.get(name.lower())
    if builder is None:
        available = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise ValueError(f"Unsupported algorithm: {name}. Available algorithms: {available}")
    return builder(config)


__all__ = ["ALGORITHM_REGISTRY", "AlgorithmBuilder", "create_algorithm"]

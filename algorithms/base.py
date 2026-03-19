from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(slots=True)
class AlgorithmConfig:
    """Runtime configuration shared by all optimization algorithms."""

    n_var: int
    n_obj: int
    xl: Sequence[float]
    xu: Sequence[float]
    pop_size: int = 100
    n_ieq_constr: int = 0
    n_gen: int = 100
    seed: int | None = None
    verbose: bool = False
    algorithm_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AlgorithmStepResult:
    """Serializable result returned after each ask-and-tell step."""

    done: bool
    generation: int
    next_x: list[list[float]] | None = None
    best_x: list[list[float]] | None = None
    best_f: list[list[float]] | None = None
    best_g: list[list[float]] | None = None


class OptimizationAlgorithm(ABC):
    """Abstract interface for optimization engines used by the API layer."""

    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config

    @abstractmethod
    def init(self) -> list[list[float]]:
        """Initialize the optimizer state and return the first candidate population."""

    @abstractmethod
    def step(
        self,
        x: Sequence[Sequence[float]],
        f: Sequence[Sequence[float]] | Sequence[float],
        g: Sequence[Sequence[float]] | Sequence[float] | None = None,
    ) -> AlgorithmStepResult:
        """Advance the optimizer using externally evaluated objective values."""

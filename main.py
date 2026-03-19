from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from algorithms import create_algorithm
from algorithms.base import AlgorithmConfig
from state_manager import get_task_manager

logger = logging.getLogger(__name__)
task_manager = get_task_manager()

app = FastAPI(
    title="Black Box Optimization Service",
    version="1.0.0",
    description="Ask-and-tell optimization service powered by FastAPI and Pymoo.",
)


class InitRequest(BaseModel):
    algorithm: str = Field(description="Algorithm name, for example ga, nsga2, or pso.")
    n_var: int = Field(gt=0, description="Number of decision variables.")
    n_obj: int = Field(default=1, gt=0, description="Number of objective values.")
    xl: float | list[float] = Field(description="Lower bound scalar or vector.")
    xu: float | list[float] = Field(description="Upper bound scalar or vector.")
    pop_size: int = Field(default=100, gt=0, description="Population size.")
    n_ieq_constr: int = Field(default=0, ge=0, description="Number of inequality constraints.")
    n_gen: int = Field(default=100, gt=0, description="Maximum generations.")
    seed: int | None = Field(default=None, description="Optional random seed.")
    verbose: bool = Field(default=False, description="Enable Pymoo verbose logging.")
    algorithm_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments passed to the selected Pymoo algorithm.",
    )


class StepRequest(BaseModel):
    task_id: str = Field(description="Task identifier returned by /init.")
    x: list[list[float]] = Field(description="Candidate solutions evaluated by the external system.")
    f: list[float] | list[list[float]] = Field(description="Objective values for X.")
    g: list[float] | list[list[float]] | None = Field(
        default=None,
        description="Optional inequality constraint values for X.",
    )


class InitResponse(BaseModel):
    task_id: str
    x: list[list[float]]


class StepResponse(BaseModel):
    task_id: str
    done: bool
    generation: int
    next_x: list[list[float]] | None = None
    best_x: list[list[float]] | None = None
    best_f: list[list[float]] | None = None
    best_g: list[list[float]] | None = None


def _normalize_bounds(bounds: float | list[float], *, n_var: int, field_name: str) -> list[float]:
    if isinstance(bounds, (int, float)):
        return [float(bounds)] * n_var

    if len(bounds) != n_var:
        raise ValueError(f"{field_name} must contain exactly {n_var} elements.")

    return [float(value) for value in bounds]


def _build_algorithm_config(request: InitRequest) -> AlgorithmConfig:
    xl = _normalize_bounds(request.xl, n_var=request.n_var, field_name="xl")
    xu = _normalize_bounds(request.xu, n_var=request.n_var, field_name="xu")

    for index, (lower, upper) in enumerate(zip(xl, xu), start=1):
        if lower > upper:
            raise ValueError(f"xl[{index}] cannot be greater than xu[{index}].")

    return AlgorithmConfig(
        n_var=request.n_var,
        n_obj=request.n_obj,
        xl=xl,
        xu=xu,
        pop_size=request.pop_size,
        n_ieq_constr=request.n_ieq_constr,
        n_gen=request.n_gen,
        seed=request.seed,
        verbose=request.verbose,
        algorithm_params=request.algorithm_params,
    )


@app.middleware("http")
async def cleanup_expired_tasks(request: Request, call_next):
    try:
        task_manager.cleanup_expired_tasks()
    except Exception:
        logger.exception("Failed to cleanup expired optimization tasks.")

    return await call_next(request)


@app.post("/init", response_model=InitResponse)
async def init_optimization(request: InitRequest) -> InitResponse:
    try:
        config = _build_algorithm_config(request)
        algorithm = create_algorithm(request.algorithm, config)
        initial_x = algorithm.init()
        task_id = task_manager.create_task(algorithm)
        return InitResponse(task_id=task_id, x=initial_x)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to initialize optimization task.")
        raise HTTPException(status_code=500, detail="Failed to initialize optimization task.") from exc


@app.post("/step", response_model=StepResponse)
async def step_optimization(request: StepRequest) -> StepResponse:
    try:
        algorithm = task_manager.get_task(request.task_id)
        result = algorithm.step(request.x, request.f, request.g)

        if result.done:
            task_manager.remove_task(request.task_id)

        return StepResponse(
            task_id=request.task_id,
            done=result.done,
            generation=result.generation,
            next_x=result.next_x,
            best_x=result.best_x,
            best_f=result.best_f,
            best_g=result.best_g,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to advance optimization task %s.", request.task_id)
        raise HTTPException(status_code=500, detail="Failed to advance optimization task.") from exc
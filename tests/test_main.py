from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from main import app
from main import InitRequest, StepRequest, init_optimization, step_optimization


class BlackBoxOptimizationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_ga_ask_and_tell_flow(self) -> None:
        init_response = await init_optimization(
            InitRequest(
                algorithm="ga",
                n_var=3,
                n_obj=1,
                xl=0,
                xu=1,
                pop_size=4,
                n_gen=2,
                seed=1,
            )
        )

        self.assertTrue(init_response.task_id)
        self.assertEqual(len(init_response.x), 4)
        self.assertEqual(len(init_response.x[0]), 3)

        first_step = await step_optimization(
            StepRequest(
                task_id=init_response.task_id,
                x=init_response.x,
                f=[1.0, 0.8, 0.6, 0.4],
            )
        )

        self.assertFalse(first_step.done)
        self.assertIsNotNone(first_step.next_x)
        self.assertIsNotNone(first_step.best_f)
        self.assertEqual(len(first_step.next_x or []), 4)

        second_step = await step_optimization(
            StepRequest(
                task_id=init_response.task_id,
                x=first_step.next_x or [],
                f=[0.9, 0.7, 0.5, 0.3],
            )
        )

        self.assertTrue(second_step.done)
        self.assertIsNone(second_step.next_x)
        self.assertIsNotNone(second_step.best_x)
        self.assertIsNotNone(second_step.best_f)
        self.assertEqual(len(second_step.best_f or []), 1)


class FrontendCommunicationTest(unittest.TestCase):
    def test_nsga3_http_flow(self) -> None:
        with TestClient(app) as client:
            init_response = client.post(
                "/init",
                json={
                    "algorithm": "nsga3",
                    "n_var": 4,
                    "n_obj": 3,
                    "xl": 0,
                    "xu": 1,
                    "pop_size": 6,
                    "n_gen": 2,
                    "seed": 1,
                    "algorithm_params": {"n_partitions": 2},
                },
            )

            self.assertEqual(init_response.status_code, 200)
            init_payload = init_response.json()
            self.assertIn("task_id", init_payload)
            self.assertEqual(len(init_payload["x"]), 6)

            first_generation_size = len(init_payload["x"])
            first_step = client.post(
                "/step",
                json={
                    "task_id": init_payload["task_id"],
                    "x": init_payload["x"],
                    "f": [
                        [float(index), float(index + 1), float(index + 2)]
                        for index in range(first_generation_size)
                    ],
                },
            )

            self.assertEqual(first_step.status_code, 200)
            first_payload = first_step.json()
            self.assertFalse(first_payload["done"])
            self.assertEqual(len(first_payload["next_x"]), 6)
            self.assertIsNotNone(first_payload["best_f"])

            second_generation_size = len(first_payload["next_x"])
            second_step = client.post(
                "/step",
                json={
                    "task_id": init_payload["task_id"],
                    "x": first_payload["next_x"],
                    "f": [
                        [float(index + 1), float(index + 2), float(index + 3)]
                        for index in range(second_generation_size)
                    ],
                },
            )

            self.assertEqual(second_step.status_code, 200)
            second_payload = second_step.json()
            self.assertTrue(second_payload["done"])
            self.assertIsNone(second_payload["next_x"])
            self.assertIsNotNone(second_payload["best_x"])
            self.assertIsNotNone(second_payload["best_f"])


if __name__ == "__main__":
    unittest.main()

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from pydantic import BaseModel
from appworld_experiments.code.gepa.gepa_agent import GEPAAgent

from gepa import EvaluationBatch, GEPAAdapter
from appworld import AppWorld
from appworld.evaluator import TestTracker

class AppWorldTask(BaseModel):
    task_id: str

class AppWorldAdapter(GEPAAdapter):

    def __init__(
        self,
        agent_config: Dict[str, Any],
        experiment_name: str,
    ):
        self.agent = GEPAAgent.from_dict(agent_config)
        self.experiment_name = experiment_name

    def evaluate(
        self,
        batch: list[AppWorldTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        outputs = []
        scores = []
        trajectories = []

        example_run_id = "_temp_gepa_run" + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
        instruct_prompt = candidate["instruction_prompt"]
        num_tasks = len(batch)
        self.agent.logger.initialize(
            experiment_name=self.experiment_name + example_run_id,
            num_tasks=num_tasks,
            num_processes=1,
            process_index=0,
        )
        self.agent.gepa_prompt_replace = instruct_prompt

        for example in batch:
            task_id = example.task_id
            test_tracker = self.agent.solve_task(task_id, self.experiment_name + example_run_id)
            try:
                success = test_tracker.success
                score = int(success)
                #score = len(test_tracker.passes) / test_tracker._num_tests
                failed_reason_list = []
                for failure in test_tracker.failures:
                    failed_reason_list.append(json.dumps(failure, indent=2))
                failed_reason = ','.join(failed_reason_list)
            except Exception as e:
                #TODO: need to handle case for failed code execution
                success = False
                score = 0
                failed_reason = "\n\n".join(test_tracker)
            outputs.append(
                f"App World outputs are omitted. Please see directory for detailed logging."
            )
            scores.append(score)
            trajectories.append(
                {
                    "messages": self.agent.messages,
                    "instruction_prompt": instruct_prompt,
                    "failed_reason": str(failed_reason),
                    "success": success,
                }
            )
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ):
        reflective_dataset = {"instruction_prompt": []}
        for score, trajectory in zip(eval_batch.scores, eval_batch.trajectories, strict=False):
            if trajectory["success"]:
                feedback = "Successfully solved the task!"
            else:
                feedback = (
                    f"Failed to solve the task. Reason: {trajectory['failed_reason']}"
                )
            reflective_dataset["instruction_prompt"].append(
                {
                    "Message History": trajectory["messages"],
                    "Instruction Prompt": candidate["instruction_prompt"],
                    "Feedback": feedback,
                }
            )
        return reflective_dataset

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from pydantic import BaseModel

from gepa import EvaluationBatch, GEPAAdapter
import io
import contextlib
import sys
from llm.src.gpt_request import collect_response_from_gpt
from llm.src.evaluation import execute_model

class BirdTask(BaseModel):
    question: str
    db_path: str
    knowledge: str
    ground_truth: str

class BirdAdapter(GEPAAdapter):

    def __init__(
        self,
        model_name: str,
    ):
        self.model_name = model_name

    def evaluate(
        self,
        batch: list[BirdTask],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        outputs = []
        scores = []
        trajectories = []

        instruct_prompt = candidate["instruction_prompt"]

        for example in batch:
            responses, message_list = collect_response_from_gpt(db_path_list=[example.db_path], question_list=[example.question], api_key="", engine=self.model_name, knowledge_list=[example.knowledge], system_message=instruct_prompt)
            predict_sql, predict_db_name = responses[0].split('\t----- bird -----\t')
            predict_db_path = ...
            ground_truth_sql = example.ground_truth.strip()
            result = execute_model(predict_sql, ground_truth_sql, predict_db_path, 0, 30.0)
            score = result['res']
            if score == 1:
                success = True
            else:
                success = False
            outputs.append("No logging here...")
            scores.append(score)
            trajectories.append(
                {
                    "messages": message_list[0],
                    "instruction_prompt": instruct_prompt,
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
                    f"Failed to solve the task."
                )
            reflective_dataset["instruction_prompt"].append(
                {
                    "Message History": trajectory["messages"],
                    "Instruction Prompt": candidate["instruction_prompt"],
                    "Feedback": feedback,
                }
            )
        return reflective_dataset

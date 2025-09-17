import os
import argparse
from appworld.common.path_store import path_store
from gepa import optimize
import litellm
from gepa.adapters.app_world_adapter.app_world_adapter import (
    AppWorldTask,
    AppWorldAdapter,
)
from appworld.common.utils import jsonnet_load, read_file
from appworld.task import Task, load_task_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="fdklsjfaskl")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    experiment_file_path = os.path.join(path_store.experiment_configs, experiment_name + ".jsonnet")
    experiment_config = jsonnet_load(
        experiment_file_path,
        APPWORLD_EXPERIMENT_PROMPTS_PATH=path_store.experiment_prompts,
        APPWORLD_EXPERIMENT_CONFIGS_PATH=path_store.experiment_configs,
        APPWORLD_EXPERIMENT_CODE_PATH=path_store.experiment_code,
    )
    runner_config = experiment_config.pop("config")
    dataset_name = runner_config.pop("dataset")
    agent_config = runner_config.pop("agent")
    code_prompt_file_path = agent_config['code_prompt_file_path']
    code_prompt_template = read_file(code_prompt_file_path.replace("/", os.sep))
    task_ids = load_task_ids(dataset_name)
    for task_id in task_ids:
        Task.load(task_id=task_id)
    trainset = [
        AppWorldTask(task_id=task_id) for task_id in task_ids[0:20]
    ]
    valset = [AppWorldTask(task_id=task_id) for task_id in task_ids[20:22]]
    testset = [AppWorldTask(task_id=task_id) for task_id in task_ids[30:40]]

    reflection_lm_name = "openai/gpt-5"
    reflection_lm = (
        lambda prompt: litellm.completion(
            model=reflection_lm_name,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
        )
        .choices[0]
        .message.content
    )

    adapter = AppWorldAdapter(agent_config, experiment_name=experiment_name)
    testset_results_before_opt = adapter.evaluate(testset, {"instruction_prompt": code_prompt_template}, capture_traces=True)

    optimized_results = optimize(
        seed_candidate={"instruction_prompt": code_prompt_template},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm="openai/gpt-5",
        max_metric_calls=5
    )

    testset_results_after_opt = adapter.evaluate(
        testset,
        {"instruction_prompt": optimized_results.best_candidate["instruction_prompt"]},
        capture_traces=True,
    )
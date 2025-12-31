import os
import argparse
from appworld.common.path_store import path_store
from gepa.core.state import GEPAState
from gepa.core.result import GEPAResult
from appworld.common.utils import jsonnet_load, read_file
from appworld.task import Task, load_task_ids
import litellm
from gepa.adapters.app_world_adapter.app_world_adapter import (
    AppWorldTask,
    AppWorldAdapter,
)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--appworld_bin_dir_path", type=str, required=True)
    parser.add_argument("--test_type", type=str, choices=["normal", "challenge"], required=True, help="Choose either normal or challenge mode.")
    args = parser.parse_args()
    experiment_name = args.experiment_name
    gepa_state_path = args.appworld_bin_dir_path
    test_type = args.test_type
    experiment_file_path = os.path.join(path_store.experiment_configs, experiment_name + ".jsonnet")
    experiment_config = jsonnet_load(
        experiment_file_path,
        APPWORLD_EXPERIMENT_PROMPTS_PATH=path_store.experiment_prompts,
        APPWORLD_EXPERIMENT_CONFIGS_PATH=path_store.experiment_configs,
        APPWORLD_EXPERIMENT_CODE_PATH=path_store.experiment_code,
    )
    runner_config = experiment_config.pop("config")
    agent_config = runner_config.pop("agent")

    gepa_state = GEPAState.load(gepa_state_path)
    result = GEPAResult.from_state(gepa_state)

    test_task_ids = load_task_ids(f'test_{test_type}')
    for task_id in test_task_ids:
        Task.load(task_id=task_id)
    print(f"Length of original test {test_type} dataset: {len(test_task_ids)}")
    testset = [AppWorldTask(task_id=task_id) for task_id in test_task_ids[:]]

    adapter = AppWorldAdapter(agent_config, experiment_name=experiment_name)
    testset_results_before_opt = adapter.evaluate(testset, {"instruction_prompt": result.best_candidate["instruction_prompt"]}, capture_traces=True)
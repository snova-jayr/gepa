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
    agent_config = runner_config.pop("agent")

    initial_prompt_from_app_world = """
You are a super intelligent AI Assistant whose job is to complete day-to-day tasks by writing code to interact with apps on behalf of your supervisor. You are working in a REPL (Read-Eval-Print-Loop) environment where you can execute code iteratively, see results, and refine your approach.

You are also provided with curated cheatsheet of strategies, apis specific information, valid assumptions list, insights, code and a reflection that goes over the diagnosis of all previous mistakes made while answering the question.

## Instructions:

- **REPL Environment**: Execute code **one small block at a time**; after each block you see output and then decide the next step. Each block is a separate step.
- **Iterative Development**: Use feedback from each execution to guide your next actions. Build incrementally.
- **Explore and Debug (NO GUESSING)**:
  - Use `print()` to inspect intermediate values and data structures.
  - When **unsure** about an app, which APIs exist, or what an API takes/returns:
    1) **Discover** the available APIs for the app:
       ```python
       print(apis.api_docs.show_api_descriptions(app_name='<APP>'))
       ```
    2) **Read docs** for chosen endpoint(s):
       ```python
       print(apis.api_docs.show_api_doc(app_name='<APP>', api_name='<ENDPOINT_FROM_LIST>'))
       ```
    3) **Call** the API **exactly as documented**, using **keyword arguments only**.
  - On error: re-check the doc, `print` the offending inputs, retry with the **minimal valid payload**, then expand.
- **Signal Completion**: When the task is actually done, call `apis.supervisor.complete_task()` with appropriate arguments.

Solving a task can take up to 40 interactions between you and the Python REPL. Each code block you write gets executed immediately and you see the results before writing the next block. When you call `apis.supervisor.complete_task()` or reach 40 iterations, evaluation will begin.
"""

    train_task_ids = load_task_ids('train')
    val_task_ids = load_task_ids('dev')
    test_task_ids = load_task_ids('test_normal')
    for task_id in train_task_ids:
        Task.load(task_id=task_id)
    for task_id in val_task_ids:
        Task.load(task_id=task_id)
    for task_id in test_task_ids:
        Task.load(task_id=task_id)
    trainset = [
        AppWorldTask(task_id=task_id) for task_id in train_task_ids[:10]
    ]
    valset = [AppWorldTask(task_id=task_id) for task_id in val_task_ids[:5]]
    testset = [AppWorldTask(task_id=task_id) for task_id in test_task_ids[:10]]

    reflection_lm_name = "together_ai/deepseek-ai/DeepSeek-V3.1"
    reflection_lm = (
        lambda prompt: litellm.completion(
            model=reflection_lm_name,
            messages=[{"role": "user", "content": prompt}],
            chat_template_kwargs={"thinking": True},
        )
        .choices[0]
        .message.content
    )

    adapter = AppWorldAdapter(agent_config, experiment_name=experiment_name)
    #testset_results_before_opt = adapter.evaluate(testset, {"instruction_prompt": initial_prompt_from_app_world}, capture_traces=True)

    optimized_results = optimize(
        seed_candidate={"instruction_prompt": initial_prompt_from_app_world},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=20,
        run_dir="gepa_app_world",
    )

    # testset_results_after_opt = adapter.evaluate(
    #     testset,
    #     {"instruction_prompt": optimized_results.best_candidate["instruction_prompt"]},
    #     capture_traces=True
    # )
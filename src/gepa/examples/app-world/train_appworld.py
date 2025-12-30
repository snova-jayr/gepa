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
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--save-price-path", type=str, default='appworld_gepa_pricing.jsonl')
    args = parser.parse_args()
    experiment_name = args.experiment_name
    price_path_file = args.save_price_path
    experiment_file_path = os.path.join(path_store.experiment_configs, experiment_name + ".jsonnet")
    experiment_config = jsonnet_load(
        experiment_file_path,
        APPWORLD_EXPERIMENT_PROMPTS_PATH=path_store.experiment_prompts,
        APPWORLD_EXPERIMENT_CONFIGS_PATH=path_store.experiment_configs,
        APPWORLD_EXPERIMENT_CODE_PATH=path_store.experiment_code,
    )
    runner_config = experiment_config.pop("config")
    agent_config = runner_config.pop("agent")
    print(f"Running with agent config: {agent_config}")

    initial_prompt_from_cleaned = """I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.

```python
print(apis.api_docs.show_app_descriptions())
```

# To get the list of apis under any app listed above, e.g. spotify

```python
print(apis.api_docs.show_api_descriptions(app_name='spotify'))
```

# To get the specification of a particular api, e.g. spotify app's login api

```python
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
```

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that I will execute, to solve the task. 

You are also provided with a curated cheatsheet of strategies, API-specific information, common mistakes, and proven solutions to help you solve the task effectively.

**Cheatsheet**: - Read the **Cheatsheet** first, then execute the task by explicitly leveraging each relevant section:
### CHEATSHEET BEGIN
## STRATEGIES AND HARD RULES
[shr-00001] Make sure to end code blocks with ``` followed by a newline(\\n).
[shr-00005] Always look at API specifications (using apis.api_docs.show_api_doc) before calling an API.
[shr-00006] Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.

## APIs TO USE FOR SPECIFIC INFORMATION
[api-00004] You can use the "supervisor" app to get information about my accounts and use the "phone" app to get information about friends and family.

## USEFUL CODE SNIPPETS AND TEMPLATES

## COMMON MISTAKES AND CORRECT STRATEGIES

## PROBLEM-SOLVING HEURISTICS AND WORKFLOWS
[psw-00002] Remember you can use the variables in your code in subsequent code blocks.
[psw-00007] Many APIs return items in "pages". Make sure to run through all the pages by looping over `page_index`.

## VERIFICATION CHECKLIST

## TROUBLESHOOTING AND PITFALLS:

## OTHERS
[misc-00003] Remember that the email addresses, access tokens and variables (e.g. spotify_password) in the example above are not valid anymore.
[misc-00008] Once you have completed the task, make sure to call apis.supervisor.complete_task(). If the task asked for some information, return it as the answer argument, i.e. call apis.supervisor.complete_task(answer=<answer>). Many tasks do not require an answer, so in those cases, just call apis.supervisor.complete_task() i.e. do not pass any argument.

### CHEATSHEET END"""

    train_task_ids = load_task_ids('train')
    val_task_ids = load_task_ids('dev')
    test_task_ids = load_task_ids('test_normal')
    for task_id in train_task_ids:
        Task.load(task_id=task_id)
    for task_id in val_task_ids:
        Task.load(task_id=task_id)
    for task_id in test_task_ids:
        Task.load(task_id=task_id)
    print(f"Length of original train dataset: {len(train_task_ids)}")
    print(f"Length of original val dataset: {len(val_task_ids)}")
    print(f"Length of original test dataset: {len(test_task_ids)}")
    trainset = [
        AppWorldTask(task_id=task_id) for task_id in train_task_ids[:]
    ]
    valset = [AppWorldTask(task_id=task_id) for task_id in val_task_ids[:]]

    gepa_prompt_gen_file = price_path_file
    with open(gepa_prompt_gen_file, "w"):
        pass
    reflection_lm_name = "sambanova/DeepSeek-V3.1"
    print(f"Running with reflector model: {reflection_lm_name}")
    def call_lm(prompt):
        response = litellm.completion(
            model=reflection_lm_name,
            messages=[{"role": "user", "content": prompt}],
        )
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        with open(gepa_prompt_gen_file, "a") as f:
            f.write(json.dumps({'input_tokens': input_tokens, 'output_tokens': output_tokens}) + "\n")
        return response.choices[0].message.content
    reflection_lm = call_lm

    adapter = AppWorldAdapter(agent_config, experiment_name=experiment_name)

    run_dir = "gepa_app_world_deepseek-v3-1"
    optimized_results = optimize(
        seed_candidate={"instruction_prompt": initial_prompt_from_cleaned},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=1434,
        display_progress_bar=True,
        run_dir=run_dir,
    )

    optimized_instruction_prompt = optimized_results.best_candidate["instruction_prompt"]

    with open(f"{run_dir}/best_prompt.txt", 'w') as f:
        f.write(optimized_instruction_prompt)
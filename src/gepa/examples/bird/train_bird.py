import os
import argparse
from gepa import optimize
import litellm
from gepa.adapters.bird_adapter.bird_adapter import (
    BirdTask,
    BirdAdapter,
)
from llm.src.gpt_request import decouple_question_schema
import json
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()
    if args.model_name is None:
        raise ValueError(f"--model_name is set to None!")
    print(f"Running with model: {args.model_name}")

    train_path = ...
    train_data = json.load(open(train_path, 'r'))
    train_db_root_path = ...
    train_question_list, train_db_path_list, train_knowledge_list, train_ground_truth_list = decouple_question_schema(datasets=train_data, db_root_path=train_db_root_path)

    eval_path = ...
    eval_data = json.load(open(eval_path, 'r'))
    eval_db_root_path = ...
    eval_question_list, eval_db_path_list, eval_knowledge_list, eval_ground_truth_list = decouple_question_schema(datasets=eval_data, db_root_path=eval_db_root_path)

    fullset = []
    for train_question, train_db_path, train_knowledge, train_ground_truth in zip(train_question_list, train_db_path_list, train_knowledge_list, train_ground_truth_list):
        fullset.append(BirdTask(question=train_question, db_path=train_db_path, knowledge=train_knowledge, ground_truth=train_ground_truth))

    random.seed(42)
    random.shuffle(fullset)
    trainset = fullset[:1000]
    valset = fullset[1000:1500]

    reflection_lm_name = args.model_name
    print(f"Running with reflector model: {reflection_lm_name}")
    reflection_lm = (
        lambda prompt: litellm.completion(
            model=reflection_lm_name,
            messages=[{"role": "user", "content": prompt}],
        )
        .choices[0]
        .message.content
    )

    adapter = BirdAdapter(model_name=reflection_lm_name)

    model_name = reflection_lm_name.split('/')[-1]

    initial_prompt_from_cleaned = "You are an expert SQL developer. Your goal is to answer the natural question and produce an SQL command that would solve the issue at hand. The user will give some information about what the tables look like and some external knowledge along with the question they want answered. Make sure the SQL query you provide is wrapped around the SQL markdown: ```sql```"

    run_dir = f"gepa_bird_{model_name}_train_1000_val_500_minibatch_20"
    optimized_results = optimize(
        seed_candidate={"instruction_prompt": initial_prompt_from_cleaned},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=4535,
        display_progress_bar=True,
        run_dir=run_dir,
        reflection_minibatch_size=20
    )

    optimized_instruction_prompt = optimized_results.best_candidate["instruction_prompt"]

    with open(f"{run_dir}/best_prompt.txt", 'w') as f:
        f.write(optimized_instruction_prompt)
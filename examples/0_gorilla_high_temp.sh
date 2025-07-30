#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/workspaces/vast-gorilla/gorillawatch/data/max_thesis/easyr1_sets/first_gorilla_parquet_dataset/train \
    data.val_files=/workspaces/vast-gorilla/gorillawatch/data/max_thesis/easyr1_sets/first_gorilla_parquet_dataset/validation \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=gorilla_high_temp \
    trainer.n_gpus_per_node=4 \
    data.rollout_batch_size=512 \
    worker.rollout.n=20 \
    data.format_prompt=./examples/format_prompt/lfw_format.jinja \
    worker.reward.reward_function=./examples/reward_function/lfw.py:compute_score \
    data.max_prompt_length=81920 \
    data.max_response_length=81920 \
    worker.rollout.temperature=1.5 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.rollout.max_num_batched_tokens=163840 \

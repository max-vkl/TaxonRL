#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=OpenGVLab/InternVL3-8B # replace it with your local file path

python3 -m verl.trainer.main \
    trainer.experiment_name=first_gorilla_run_intern \
    config=examples/config.yaml \
    data.train_files=/workspaces/vast-gorilla/gorillawatch/data/max_thesis/easyr1_sets/first_gorilla_parquet_dataset/train \
    data.val_files=/workspaces/vast-gorilla/gorillawatch/data/max_thesis/easyr1_sets/first_gorilla_parquet_dataset/validation \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=4 \
    data.rollout_batch_size=512 \
    worker.rollout.n=20 \
    data.format_prompt=./examples/format_prompt/lfw_format.jinja \
    worker.reward.reward_function=./examples/reward_function/lfw.py:compute_score \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    worker.actor.model.trust_remote_code=true

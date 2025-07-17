#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=maxvonk/gorilla-pairs-v1@train \
    data.val_files=maxvonk/gorilla-pairs-v1@validation \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=first_gorilla_run \
    trainer.n_gpus_per_node=4 \
    data.rollout_batch_size=512 \
    worker.rollout.n=20 \
    data.format_prompt=/workspaces/EasyR1/examples/format_prompt/lfw_format.jinja \
    worker.reward.reward_function=/workspaces/EasyR1/examples/reward_function/lfw.py:compute_score \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    worker.actor.micro_batch_size_per_device_for_update=2 \

#data.train_files=/workspaces/vast-gorilla/gorillawatch/data/max_thesis/easyr1_sets/first_gorilla_parquet_dataset/train \
#data.val_files=/workspaces/vast-gorilla/gorillawatch/data/max_thesis/easyr1_sets/first_gorilla_parquet_dataset/validation \
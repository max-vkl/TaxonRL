#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    trainer.experiment_name=1.4_basic_v3_10k \
    config=examples/config.yaml \
    data.train_files=maxvonk/gorilla-pairs-v3-10k-800px@train \
    data.val_files=maxvonk/gorilla-pairs-v3-10k-800px@validation \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=4 \
    data.rollout_batch_size=128 \
    worker.rollout.n=20 \
    data.format_prompt=/workspaces/EasyR1/examples/format_prompt/lfw_format.jinja \
    worker.reward.reward_function=/workspaces/EasyR1/examples/reward_function/lfw.py:compute_score \
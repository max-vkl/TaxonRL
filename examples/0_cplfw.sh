#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=maxvonk/cplfw_normal@train \
    data.val_files=maxvonk/cplfw_normal@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=first_cplfw_test \
    trainer.n_gpus_per_node=4 \
    data.rollout_batch_size=128 \
    worker.rollout.n=16 \
    data.format_prompt=./examples/format_prompt/lfw_format.jinja \
    worker.reward.reward_function=./examples/reward_function/lfw.py:compute_score

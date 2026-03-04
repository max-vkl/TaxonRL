#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    trainer.experiment_name=3.2_whale_6g_192b_30r_test \
    config=examples/config.yaml \
    data.train_files=maxvonk/whale-pairs-v1-800px@train \
    data.val_files=maxvonk/whale-pairs-v1-800px@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=6 \
    worker.actor.global_batch_size=192 \
    data.rollout_batch_size=192 \
    worker.rollout.n=30 \
    data.format_prompt=/workspaces/EasyR1/examples/format_prompt/lfw_format.jinja \
    worker.reward.reward_function=/workspaces/EasyR1/examples/reward_function/lfw.py:compute_score \
    trainer.save_limit=40 \
    trainer.save_freq=40 \
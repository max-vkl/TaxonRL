#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/workspaces/EasyR1/start_checkpoints/qwen-7b-gemini-1k-traces-s120 # replace it with your local file path

python3 -m verl.trainer.main \
    trainer.experiment_name=1.8_basic_v3_15k_gemini_ct_6g \
    config=examples/config.yaml \
    data.train_files=maxvonk/gorilla-pairs-v3-15k-800px-sft-gemini-flash-1k@train \
    data.val_files=maxvonk/gorilla-pairs-v3-15k-800px-sft-gemini-flash-1k@validation \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=8 \
    worker.actor.global_batch_size=192 \
    data.rollout_batch_size=192 \
    worker.rollout.n=20 \
    data.format_prompt=/workspaces/EasyR1/examples/format_prompt/lfw_format.jinja \
    worker.reward.reward_function=/workspaces/EasyR1/examples/reward_function/lfw_relax_format.py:compute_score \
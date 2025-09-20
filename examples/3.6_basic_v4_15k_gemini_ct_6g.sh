#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/workspaces/EasyR1/start_checkpoints/qwen-7b-gemini-1k-traces-s120 # replace it with your local file path

python3 -m verl.trainer.main \
    trainer.experiment_name=3.6_basic_v4_15k_gemini_ct_6g \
    config=examples/config.yaml \
    data.train_files=maxvonk/gorilla-pairs-v4-800px-cv-im@train \
    data.val_files=maxvonk/gorilla-pairs-v4-800px-cv-im@validation \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=6 \
    worker.actor.global_batch_size=96 \
    data.rollout_batch_size=96 \
    worker.rollout.n=16 \
    data.format_prompt=/workspaces/EasyR1/examples/format_prompt/lfw_format.jinja \
    worker.reward.reward_function=/workspaces/EasyR1/examples/reward_function/lfw_relax_format.py:compute_score \
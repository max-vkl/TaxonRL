#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    trainer.experiment_name=3.3_chimp_clean_6g_diff_test \
    config=examples/config.yaml \
    data.train_files=maxvonk/chimp-pairs-v2-800px-meta-data-fixed-names@train \
    data.val_files=maxvonk/chimp-pairs-v2-800px-meta-data-fixed-names@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=6 \
    worker.actor.global_batch_size=96 \
    data.rollout_batch_size=96 \
    worker.rollout.n=16 \
    data.format_prompt=/workspaces/EasyR1/examples/format_prompt/lfw_format.jinja \
    worker.reward.reward_function=/workspaces/EasyR1/examples/reward_function/lfw.py:compute_score \
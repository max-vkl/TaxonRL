#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    trainer.experiment_name=3.5_bird_paper_im_concrete_3b_4g \
    config=examples/config.yaml \
    data.train_files=maxvonk/bird-pairs-v2-im-800px@train \
    data.val_files=maxvonk/bird-pairs-v2-im-800px@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=4 \
    worker.actor.global_batch_size=96 \
    data.rollout_batch_size=96 \
    worker.rollout.n=16 \
    data.format_prompt=/workspaces/EasyR1/examples/format_prompt/birds_im_concrete.jinja \
    worker.reward.reward_function=/workspaces/EasyR1/examples/reward_function/lfw_bird_im_concrete.py:compute_score_gemini \
    trainer.find_last_checkpoint=false \
    trainer.save_limit=40 \
    trainer.save_freq=40 \
    data.answer_key=answer_im \
    trainer.total_epochs=60 \
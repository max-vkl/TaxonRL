#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/workspaces/EasyR1/start_checkpoints/bird-sft-7b-full-unfrozen/checkpoint-1920 # replace it with your local file path

python3 -m verl.trainer.main \
    trainer.experiment_name=3.7_bird_paper_sft_im_concrete_6g \
    config=examples/config.yaml \
    data.train_files=maxvonk/bird-pairs-v2-im-800px@train \
    data.val_files=maxvonk/bird-pairs-v2-im-800px@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=6 \
    worker.actor.global_batch_size=192 \
    data.rollout_batch_size=192 \
    worker.rollout.n=30 \
    data.format_prompt=/workspaces/EasyR1/examples/format_prompt/birds_im_concrete_better.jinja \
    worker.reward.reward_function=/workspaces/EasyR1/examples/reward_function/lfw_bird_im_concrete.py:compute_score_gemini \
    trainer.find_last_checkpoint=false \
    trainer.save_limit=40 \
    trainer.save_freq=40 \
    data.answer_key=answer_im \
    trainer.total_epochs=60 \
#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct # replace it with your local file path

python3 -m verl.trainer.main \
    trainer.experiment_name=4.0_sea_star_im_6g \
    config=examples/config.yaml \
    data.train_files=maxvonk/sea-star-pairs-800px@train \
    data.val_files=maxvonk/sea-star-pairs-800px@train \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=6 \
    worker.actor.global_batch_size=48 \
    data.rollout_batch_size=48 \
    worker.rollout.n=16 \
    data.format_prompt=/workspaces/EasyR1/examples/format_prompt/rebuttal_sea_star_im.jinja \
    worker.reward.reward_function=/workspaces/EasyR1/examples/reward_function/lfw_sea_star_im.py:compute_score_gemini \
    trainer.find_last_checkpoint=true \
    trainer.save_limit=40 \
    trainer.save_freq=40 \
    data.answer_key=answer_im \
    trainer.total_epochs=60 \
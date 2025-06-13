#!/usr/bin/env python
# encoding: utf-8
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
import torch
import fire
from collections import defaultdict
import os


def main(fsdp_checkpoint_path, huggingface_model_path, output_path):
    # Merge FSDP checkpoints
    state_dict = defaultdict(list)
    world_size = 4  # Adjust this if your world_size is different

    for rank in range(world_size):
        filepath = os.path.join(
            fsdp_checkpoint_path, f"model_world_size_{world_size}_rank_{rank}.pt"
        )
        print("Loading", filepath)
        this_state_dict = torch.load(filepath, map_location="cpu")
        for key, value in this_state_dict.items():
            if hasattr(value, "to_local"):  # For FSDP-wrapped shards
                value = value.to_local()
            state_dict[key].append(value)

    # Concatenate sharded tensors
    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    # Load config and initialize model
    config = AutoConfig.from_pretrained(huggingface_model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.load_state_dict(state_dict)

    # Save model
    model.save_pretrained(output_path, max_shard_size="10GB")

    # Save tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(
        huggingface_model_path, trust_remote_code=True
    )
    tokenizer.save_pretrained(output_path)

    processor = AutoProcessor.from_pretrained(
        huggingface_model_path, trust_remote_code=True
    )
    processor.save_pretrained(output_path)

    print("Model, tokenizer, and processor saved to", output_path)


if __name__ == "__main__":
    fire.Fire(main)

# python convert_fsdp_to_hf.py /workspaces/vast-gorilla/gorillawatch/max_thesis/easyr1_checkpoints/first_lfw_test/global_step_30/actor/ /workspaces/vast-gorilla/gorillawatch/max_thesis/easyr1_checkpoints/first_lfw_test/global_step_30/actor/huggingface/ qwen2.5_7b_rl_step30

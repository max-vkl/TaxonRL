import argparse
import re
from io import BytesIO
from typing import Any, Dict, List

import numpy as np
import requests
from PIL import Image
from urllib.parse import urlparse
from unittest.mock import patch

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from jinja2 import Template
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

# Notiz: Mit dem vllm 0.8.4, dass in dem container ist gibt es buggs, deshalb auch 0.8.5 upgraden


def format_reward(predict: str) -> float:
    pattern = re.compile(
        r"^<think>.*?</think>\s*<answer>\d+\.\d{4}</answer>$", re.DOTALL
    )
    return 1.0 if pattern.fullmatch(predict) else 0.0


def binary_cross_entropy_reward(predict: str, ground_truth) -> float:
    # print(f"DEBUG: predict={predict!r}, ground_truth={ground_truth!r}")
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict)
        given_answer = (
            content_match.group(1).strip() if content_match else predict.strip()
        )
        # print(f"DEBUG: extracted_answer={given_answer!r}")

        # Handle ground_truth properly based on its type
        if isinstance(ground_truth, (int, float)):
            true_prob = float(ground_truth)
        else:
            true_prob = float(ground_truth.strip())

        # Try to convert prediction to float
        try:
            given_prob = float(given_answer)
            # print(f"DEBUG: given_prob={given_prob}, true_prob={true_prob}")

            # Ensure values are between 0 and 1
            if 0 <= given_prob <= 1 and 0 <= true_prob <= 1:
                # Calculate binary cross entropy
                epsilon = 1e-15  # Small value to avoid log(0)
                given_prob = max(min(given_prob, 1 - epsilon), epsilon)
                bce = -(
                    true_prob * np.log(given_prob)
                    + (1 - true_prob) * np.log(1 - given_prob)
                )
                reward = 1 - min(bce, 1)
                # print(f"DEBUG: BCE={bce}, reward={reward}")
                return reward
            else:
                # print(f"DEBUG: Values not in range 0-1")
                pass
        except ValueError as e:
            # For non-numeric predictions, fall back to exact match
            # Convert ground truth to string for comparison
            gt_str = str(ground_truth).strip()
            exact_match = given_answer == gt_str
            # print(f"DEBUG: ValueError={e}, exact_match={exact_match}")
            return 1.0 if exact_match else 0.0

    except Exception as e:
        # print(f"DEBUG: Exception={e}")
        pass

    return 0.0


def compute_score(
    predicts: List[str], ground_truths: List[str], format_weight: float = 0.5
) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(
            r"\s*(<|>|/)\s*", r"\1", predict
        )  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = binary_cross_entropy_reward(predict, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score
                + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores


def process_image(img_data: Any) -> Image.Image:
    """
    Load an image from HTTP(S), file:// URI, local path, or bytes.
    """
    img: Image.Image
    if isinstance(img_data, Image.Image):
        return img_data.convert("RGB")
    if isinstance(img_data, dict) and "bytes" in img_data:
        img_data = img_data["bytes"]
    if isinstance(img_data, bytes):
        img = Image.open(BytesIO(img_data))
    else:  # assume it is a URL or path
        path_or_url = str(img_data)
        parsed = urlparse(path_or_url)
        if parsed.scheme in ("http", "https"):
            resp = requests.get(path_or_url)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content))
        elif parsed.scheme == "file":
            img = Image.open(parsed.path)
        else:
            img = Image.open(path_or_url)
    return img.convert("RGB")


def build_request(sample, processor, prompt_template: Template, image_key: str, problem_key: str):
    images_data = sample[image_key]
    if not isinstance(images_data, list):
        images_data = [images_data]

    imgs = [process_image(u) for u in images_data]
    question = sample[problem_key]
    if prompt_template:
        question = prompt_template.render(content=question)

    placeholders = [{"type": "image", "image": None} for _ in imgs]
    user_content = [*placeholders, {"type": "text", "text": question}]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": imgs},
    }


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--ckpt", type=str, default="qwen2.5_7b_rl_step30")
    parser.add_argument("--ckpt", type=str, default="Qwen/qwen2.5-vl-7b-instruct")
    parser.add_argument("--dataset", type=str, default="maxvonk/lfw")
    parser.add_argument("--image-key", type=str, default="images")
    parser.add_argument("--problem-key", type=str, default="problem")
    parser.add_argument("--answer-key", type=str, default="answer")
    parser.add_argument(
        "--format-prompt-path", type=str, default="/workspaces/EasyR1/examples/format_prompt/lfw_format.jinja"
    )
    args = parser.parse_args()

    # Load prompt template
    with open(args.format_prompt_path) as f:
        prompt_template = Template(f.read())

    # 1) Load model & processor conditionally
    if args.ckpt.startswith("Qwen/"):
        # official HuggingFace model
        model_name = args.ckpt
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
    else:
        # local fine-tuned checkpoint
        model_name = args.ckpt
        _ = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            use_cache=False,
        )
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left")

    # 2) Instantiate VLLM runner with single-GPU patches
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        vlm = LLM(
            model=model_name,
            # if using local model object, vllm will pick it up by name_or_path
            device="cuda:0",
            gpu_memory_utilization=0.9,
            dtype="bfloat16",
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 10, "video": 0},
        )

    # 3) Load dataset
    if "@" in args.dataset:
        dataset_name, dataset_split = args.dataset.split("@")
    else:
        dataset_name = args.dataset
        dataset_split = "validation"
    dataset = load_dataset(dataset_name, split=dataset_split)

    if not isinstance(dataset, Dataset):
        print("Dataset is not a standard indexable dataset, skipping evaluation.")
        return

    all_predictions = []
    all_ground_truths = []
    # 4) Inference in batches of 25
    batch_size = 100
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        # batch is a dict of lists, need to convert to list of dicts
        keys = list(batch.keys())
        requests_batch = [
            build_request(
                {k: batch[k][i] for k in keys},
                processor,
                prompt_template,
                args.image_key,
                args.problem_key,
            )
            for i in range(len(batch[keys[0]]))
        ]
        outputs = vlm.generate(requests_batch, sampling_params=sampling_params)
        
        predictions = [output.outputs[0].text for output in outputs]
        ground_truths = batch[args.answer_key]

        all_predictions.extend(predictions)
        all_ground_truths.extend(ground_truths)

    scores = compute_score(all_predictions, all_ground_truths)
    if scores:
        accuracy_scores = [s["accuracy"] for s in scores]
        average_accuracy = np.mean(accuracy_scores)
        print(f"Final Average Accuracy: {average_accuracy}")


if __name__ == "__main__":
    main()

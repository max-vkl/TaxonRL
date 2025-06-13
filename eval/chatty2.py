import argparse
import requests
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse
from unittest.mock import patch

from vllm import LLM, SamplingParams
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def fetch_image(url: str) -> Image.Image:
    """
    Load an image from HTTP(S), file:// URI, or local path.
    """
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        resp = requests.get(url)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
    elif parsed.scheme == "file":
        img = Image.open(parsed.path)
    else:
        img = Image.open(url)
    return img.convert("RGB")


def build_request(sample, processor):
    imgs = [fetch_image(u) for u in sample["image_urls"]]
    placeholders = [{"type": "image", "image": None} for _ in imgs]
    user_content = [*placeholders, {"type": "text", "text": sample["question"]}]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": imgs},
    }


def main():
    ckpt = "qwen2.5_7b_rl_step30"

    # 1) Load model & processor conditionally
    if ckpt.startswith("Qwen/"):
        # official HuggingFace model
        model_name = ckpt
        model = None  # vllm will load by name
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
    else:
        # local fine-tuned checkpoint
        model_name = ckpt
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            use_cache=False,
        )
        model.eval()
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
            gpu_memory_utilization=1.0,
            dtype="bfloat16",
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 2, "video": 0},
        )

    # 3) Build a 200-sample dataset
    image_urls = [
        "file:///workspaces/EasyR1/assets/wechat.jpg",
        "file:///workspaces/EasyR1/assets/wechat.jpg",
    ]
    dataset = [
        {
            "question": "Tell me the difference between these two images.",
            "image_urls": image_urls,
        }
        for _ in range(200)
    ]

    # 4) Inference in batches of 25
    batch_size = 25
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
        requests_batch = [build_request(s, processor) for s in batch]
        outputs = vlm.generate(requests_batch, sampling_params=sampling_params)


if __name__ == "__main__":
    main()

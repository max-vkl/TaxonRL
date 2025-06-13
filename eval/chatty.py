import requests
from io import BytesIO
from PIL import Image
from unittest.mock import patch

from vllm import LLM, SamplingParams
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import torch.distributed as dist
from urllib.parse import urlparse


def fetch_image(url: str) -> Image.Image:
    """
    Load an image either from HTTP(S) or from a local file:// URI (or plain path).
    """
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        resp = requests.get(url)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
    elif parsed.scheme == "file":
        # file:///absolute/path/to/img.png  →  /absolute/path/to/img.png
        img = Image.open(parsed.path)
    else:
        # treat anything else as a plain filesystem path
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
    # 1) Load checkpoint into HF model & processor
    checkpoint = "qwen2.5_7b_rl_step30"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype="auto",
        use_cache=False,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(checkpoint, padding_side="left")

    # 2) Instantiate VLLM runner (single-GPU patches)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        vlm = LLM(
            model=model.name_or_path,
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

    # 4) Run inference in batches of 25
    batch_size = 5
    for batch_start in range(0, len(dataset), batch_size):
        batch_samples = dataset[batch_start : batch_start + batch_size]
        requests_batch = [build_request(s, processor) for s in batch_samples]

        outputs = vlm.generate(requests_batch, sampling_params=sampling_params)

    # 5) Print each result
    for i, out in enumerate(outputs, start=batch_start + 1):
        text = out.outputs[0].text.strip()
        # print(f"--- Sample {i} ---")
        # print(text)
        # print()


if __name__ == "__main__":
    main()

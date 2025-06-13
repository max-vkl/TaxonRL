import re
import numpy as np
from typing import Dict, List

from mathruler.grader import grade_answer


# def format_reward(predict: str) -> float:
#     pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
#     format_match = re.fullmatch(pattern, predict)
#     return 1.0 if format_match else 0.0


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

import re
import numpy as np
from typing import Dict, List, Any

from mathruler.grader import grade_answer


def format_reward(predict: str) -> float:
    pattern = re.compile(
        r"^<think>.*?</think>\s*<answer>\d+\.\d{4}</answer>$", re.DOTALL
    )
    return 1.0 if pattern.fullmatch(predict) else 0.0


def binary_cross_entropy_reward(predict: str, ground_truth) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict)
        given_answer = (
            content_match.group(1).strip() if content_match else predict.strip()
        )

        if isinstance(ground_truth, (int, float)):
            true_prob = float(ground_truth)
        else:
            true_prob = float(ground_truth.strip())

        try:
            given_prob = float(given_answer)

            if 0 <= given_prob <= 1 and 0 <= true_prob <= 1:
                epsilon = 1e-15
                given_prob = max(min(given_prob, 1 - epsilon), epsilon)
                bce = -(
                    true_prob * np.log(given_prob)
                    + (1 - true_prob) * np.log(1 - given_prob)
                )
                reward = 1 - min(bce, 1)
                return reward
        except ValueError:
            gt_str = str(ground_truth).strip()
            exact_match = given_answer == gt_str
            return 1.0 if exact_match else 0.0

    except Exception:
        pass

    return 0.0


def precision_f1_reward(predict: str, ground_truth) -> tuple[float, float, float]:
    """
    Computes per-sample precision and F1 for the final answer probability using a 0.5 threshold.
    Also computes acutal_accuracy as threshold-based correctness (pred class equals true class).
    Returns (precision, f1, acutal_accuracy). If parsing fails or values are out of [0, 1], returns (0.0, 0.0, 0.0).
    """
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict)
        given_answer = (
            content_match.group(1).strip() if content_match else predict.strip()
        )

        if isinstance(ground_truth, (int, float)):
            true_prob = float(ground_truth)
        else:
            true_prob = float(str(ground_truth).strip())

        given_prob = float(given_answer)

        if 0 <= given_prob <= 1 and 0 <= true_prob <= 1:
            threshold = 0.5
            pred_pos = given_prob >= threshold
            true_pos = true_prob >= threshold

            tp = 1 if pred_pos and true_pos else 0
            fp = 1 if pred_pos and not true_pos else 0
            fn = 1 if (not pred_pos) and true_pos else 0
            tn = 1 if (not pred_pos) and (not true_pos) else 0

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            acutal_accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            return precision, f1, acutal_accuracy
    except Exception:
        pass

    return 0.0, 0.0, 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for lfw reward function.")

    scores = []
    total_true_positive = 0
    total_true_negative = 0
    total_false_positive = 0
    total_false_negative = 0
    for reward_input in reward_inputs:
        predict = re.sub(
            r"\s*(<|>|/)\s*", r"\1", reward_input["response"]
        )  # handle qwen2.5vl-32b format
        ground_truth = reward_input["ground_truth"]
        format_score = format_reward(predict)
        accuracy_score = binary_cross_entropy_reward(predict, ground_truth)
        precision_score, f1_score, acutal_accuracy = precision_f1_reward(predict, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score
                + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "precision": precision_score,
                "f1": f1_score,
                "acutal_accuracy": acutal_accuracy,
            }
        )

        # Accumulate confusion matrix components for macro metrics
        try:
            content_match = re.search(r"<answer>(.*?)</answer>", predict)
            given_answer_text = (
                content_match.group(1).strip() if content_match else predict.strip()
            )

            if isinstance(ground_truth, (int, float)):
                true_prob = float(ground_truth)
            else:
                true_prob = float(str(ground_truth).strip())

            given_prob = float(given_answer_text)

            if 0 <= given_prob <= 1 and 0 <= true_prob <= 1:
                threshold = 0.5
                pred_pos = given_prob >= threshold
                true_pos = true_prob >= threshold

                if pred_pos and true_pos:
                    total_true_positive += 1
                elif pred_pos and not true_pos:
                    total_false_positive += 1
                elif (not pred_pos) and true_pos:
                    total_false_negative += 1
                else:
                    total_true_negative += 1
        except Exception:
            pass

    # Compute macro metrics over the whole batch and overwrite per-item placeholders
    precision_denominator = total_true_positive + total_false_positive
    recall_denominator = total_true_positive + total_false_negative
    accuracy_denominator = (
        total_true_positive + total_true_negative + total_false_positive + total_false_negative
    )

    macro_precision = (
        total_true_positive / precision_denominator if precision_denominator > 0 else 0.0
    )
    macro_recall = total_true_positive / recall_denominator if recall_denominator > 0 else 0.0
    macro_f1 = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    )
    macro_accuracy = (
        (total_true_positive + total_true_negative) / accuracy_denominator
        if accuracy_denominator > 0
        else 0.0
    )

    for item in scores:
        item["precision"] = macro_precision
        item["f1"] = macro_f1
        item["macro_accuracy"] = macro_accuracy

    return scores


def parse_ground_truth_fields(ground_truth: str) -> tuple[str, str, str]:
    """
    Sea star layout:
      - 'answer;species1;species2'  where species are abbreviations:
        Asru = Asterias rubens, Anau = Anthenea australiae
      Returns (answer, species1, species2).
    """
    parts = [p.strip() for p in str(ground_truth).split(";")]
    main_gt = parts[0] if len(parts) >= 1 else ""

    # Sea stars (>=3 fields): pick indices 1 and 2
    if len(parts) >= 3:
        f1 = parts[1]
        f2 = parts[2]
    else:
        f1 = ""
        f2 = ""

    return main_gt, f1, f2


def _canonicalize_species(label: str) -> str:
    """
    Map a variety of label forms to canonical abbreviations:
      - 'Asru' or any variant of 'Asterias rubens' -> 'Asru'
      - 'Anau' or any variant of 'Anthenea australiae' -> 'Anau'
    """
    s = label.strip().lower()
    # Normalize punctuation/spaces like "A. rubens"
    s_nopunct = re.sub(r"[.\s]+", " ", s).strip()

    # Direct abbreviation matches
    if s_nopunct in {"asru"}:
        return "Asru"
    if s_nopunct in {"anau"}:
        return "Anau"

    # Name-based heuristics
    if "asterias" in s_nopunct or " rubens" in s_nopunct or s_nopunct.endswith(" rubens"):
        return "Asru"
    if "anthenea" in s_nopunct or " australiae" in s_nopunct or s_nopunct.endswith(" australiae"):
        return "Anau"

    # Fallback to original (title-cased) to avoid false positives; will likely not match GT
    return label.strip()


def species_reward(
    predict: str, gt_species1: str, gt_species2: str, species_reward_format: str
) -> tuple[float, bool]:
    """
    Computes a score based on correctness of predicted species.
    Accepts <species>…</species> with either:
      - chatgpt: <species>Image1=...; Image2=...</species>
      - gemini:  <species>...; ...</species>
    Scores 1.0 for two correct, 0.5 for one correct, 0.0 for none.
    GT species are abbreviations: 'Asru' or 'Anau'. Predictions may use full names or abbreviations.
    """
    if species_reward_format == "chatgpt":
        # e.g. <species>Image1=Asterias rubens; Image2=Anthenea australiae</species>
        match = re.search(
            r"<species>Image1=([^;]+);\s*Image2=([^<]+)</species>", predict
        )
    elif species_reward_format == "gemini":
        # e.g. <species>Asterias rubens; Anthenea australiae</species>
        match = re.search(
            r"<species>([^;]+);\s*([^<]+)</species>", predict
        )
    else:
        raise ValueError(f"Invalid species_reward_format: {species_reward_format}")

    if not match:
        return 0.0, False

    pred_1_raw = match.group(1).strip()
    pred_2_raw = match.group(2).strip()

    pred_1 = _canonicalize_species(pred_1_raw)
    pred_2 = _canonicalize_species(pred_2_raw)

    # Canonicalize GT as well (in case it's already full name for some rows)
    gt_1 = _canonicalize_species(str(gt_species1))
    gt_2 = _canonicalize_species(str(gt_species2))

    correct_count = 0
    if pred_1 == gt_1:
        correct_count += 1
    if pred_2 == gt_2:
        correct_count += 1

    return correct_count * 0.5, True


def compute_score_chatgpt(
    reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for lfw reward function.")

    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        raw_ground_truth = reward_input["ground_truth"]
        main_gt, gt_aux1, gt_aux2 = parse_ground_truth_fields(raw_ground_truth)

        format_score = format_reward(predict)
        accuracy_score = binary_cross_entropy_reward(predict, main_gt)
        # Species correctness score from the <species> tag
        species_score, _ = species_reward(predict, gt_aux1, gt_aux2, "chatgpt")
        precision_score, f1_score, acutal_accuracy = precision_f1_reward(predict, main_gt)

        overall = 0.4 * accuracy_score + 0.4 * species_score + format_weight * format_score
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
                "species": species_score,
                "precision": precision_score,
                "f1": f1_score,
                "acutal_accuracy": acutal_accuracy,
            }
        )

    return scores


def compute_score_gemini(
    reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for lfw reward function.")

    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        raw_ground_truth = reward_input["ground_truth"]
        main_gt, gt_aux1, gt_aux2 = parse_ground_truth_fields(raw_ground_truth)

        format_score = format_reward(predict)
        accuracy_score = binary_cross_entropy_reward(predict, main_gt)
        species_score, _ = species_reward(predict, gt_aux1, gt_aux2, "gemini")
        precision_score, f1_score, acutal_accuracy = precision_f1_reward(predict, main_gt)

        overall = ((1 - format_weight) / 2) * accuracy_score + ((1 - format_weight) / 2) * species_score + format_weight * format_score
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
                "species": species_score,
                "precision": precision_score,
                "f1": f1_score,
                "acutal_accuracy": acutal_accuracy,
            }
        )

    return scores

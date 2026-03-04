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


def parse_ground_truth_fields(
    ground_truth: str,
) -> tuple[
    str,  # main ground truth (e.g., similarity probability or label)
    str,  # actual_distance (kept for completeness)
    str,  # img1Genus
    str,  # img1Family
    str,  # img1Order
    str,  # img2Genus
    str,  # img2Family
    str,  # img2Order
]:
    """
    Parses bird taxonomy ground truth fields formatted as:
    'ground_truth;actual_distance;img1Genus;img1Family;img1Order;img2Genus;img2Family;img2Order'
    """
    parts = [p.strip() for p in str(ground_truth).split(";")]

    main_gt = parts[0] if len(parts) >= 1 else ""
    actual_distance = parts[1] if len(parts) >= 2 else ""
    img1_genus = parts[2] if len(parts) >= 3 else ""
    img1_family = parts[3] if len(parts) >= 4 else ""
    img1_order = parts[4] if len(parts) >= 5 else ""
    img2_genus = parts[5] if len(parts) >= 6 else ""
    img2_family = parts[6] if len(parts) >= 7 else ""
    img2_order = parts[7] if len(parts) >= 8 else ""

    return (
        main_gt,
        actual_distance,
        img1_genus,
        img1_family,
        img1_order,
        img2_genus,
        img2_family,
        img2_order,
    )


def taxonomic_intermediate_reward(
    predict: str,
    img1_order: str,
    img2_order: str,
    img1_family: str,
    img2_family: str,
    img1_genus: str,
    img2_genus: str,
) -> tuple[float, bool]:
    """
    Scores whether the model's taxonomic conclusions within <think> match ground-truth equality at
    Order, Family, and Genus levels. The prompt requires tags <order>, <family>, <genus> whose
    values are one of: Same, Different, Uncertain.

    Reward: average of per-level correctness over present tags.
    - Correct if tag matches whether GT labels are equal at that level.
    - Uncertain or missing tag is scored as incorrect (0) for that level.
    Returns (score in [0,1], found_any_tag).
    """

    def normalize_taxon(value: str) -> str:
        return (value or "").strip().lower()

    def tags_equal(gt_a: str, gt_b: str) -> bool | None:
        a = normalize_taxon(gt_a)
        b = normalize_taxon(gt_b)
        if not a or not b:
            return None
        return a == b

    def extract_level_decision(text: str, tag: str) -> str | None:
        m = re.search(rf"<{tag}>\s*(Same|Different|Uncertain)\s*</{tag}>", text, re.IGNORECASE)
        return m.group(1).capitalize() if m else None

    levels = [
        ("order", tags_equal(img1_order, img2_order)),
        ("family", tags_equal(img1_family, img2_family)),
        ("genus", tags_equal(img1_genus, img2_genus)),
    ]

    scores: list[float] = []
    found_any = False
    for level_name, gt_same in levels:
        decision = extract_level_decision(predict, level_name)
        if decision is None:
            continue
        found_any = True
        if gt_same is None:
            # GT missing; skip scoring this level
            continue
        if decision == "Same":
            scores.append(1.0 if gt_same else 0.0)
        elif decision == "Different":
            scores.append(1.0 if (not gt_same) else 0.0)
        else:  # Uncertain
            scores.append(0.0)

    if not scores:
        return 0.0, False

    return float(sum(scores) / len(scores)), found_any

def taxonomy_reward_from_distance(predict: str, actual_distance: str) -> float:
    """
    Scores taxonomic tags based on the provided actual_distance label, ignoring concrete taxonomy.

    actual_distance in {sameSpecies, sameGenus, sameFamily, sameOrder, sameClass} (case-insensitive)

    Expected tag decisions:
      - sameSpecies  -> order=Same, family=Same, genus=Same
      - sameGenus    -> order=Same, family=Same, genus=Same
      - sameFamily   -> order=Same, family=Same, genus=Different
      - sameOrder    -> order=Same, family=Different, genus=Different
      - sameClass    -> order=Different, family=Different, genus=Different

    Scoring: Each of the three tags contributes equally (1/3). Missing tags are treated as "Different".
    """
    dist = (actual_distance or "").strip().lower()

    if dist in ("samespecies", "samegenus"):
        expected = {"order": "Same", "family": "Same", "genus": "Same"}
    elif dist == "samefamily":
        expected = {"order": "Same", "family": "Same", "genus": "Different"}
    elif dist == "sameorder":
        expected = {"order": "Same", "family": "Different", "genus": "Different"}
    else:  # sameClass or anything else defaults to only class same → all below different
        expected = {"order": "Different", "family": "Different", "genus": "Different"}

    def extract_decision(text: str, tag: str) -> str:
        m = re.search(rf"<{tag}>\s*(Same|Different|Uncertain)\s*</{tag}>", text, re.IGNORECASE)
        if not m:
            # Treat missing tag as Different
            return "Different"
        return m.group(1).capitalize()

    decisions = {
        "order": extract_decision(predict, "order"),
        "family": extract_decision(predict, "family"),
        "genus": extract_decision(predict, "genus"),
    }

    correct = 0
    for level in ("order", "family", "genus"):
        if decisions[level] == expected[level]:
            correct += 1
        else:
            # Uncertain or mismatched counts as incorrect
            pass

    return correct / 3.0

def compute_score_gemini(
    reward_inputs: List[Dict[str, Any]], format_weight: float = 0.2
) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for lfw reward function.")

    scores = []
    for reward_input in reward_inputs:
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # normalize tag spacing
        raw_ground_truth = reward_input["ground_truth"]
        (
            main_gt,
            actual_distance,
            _img1_genus,
            _img1_family,
            _img1_order,
            _img2_genus,
            _img2_family,
            _img2_order,
        ) = parse_ground_truth_fields(raw_ground_truth)

        format_score = format_reward(predict)
        accuracy_score = binary_cross_entropy_reward(predict, main_gt)
        taxonomy_score = taxonomy_reward_from_distance(predict, actual_distance)
        precision_score, f1_score, acutal_accuracy = precision_f1_reward(predict, main_gt)

        overall = (
            ((1 - format_weight) / 2) * accuracy_score
            + ((1 - format_weight) / 2) * taxonomy_score
            + format_weight * format_score
        )
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
                "taxonomy": taxonomy_score,
                "precision": precision_score,
                "f1": f1_score,
                "acutal_accuracy": acutal_accuracy,
            }
        )

    return scores

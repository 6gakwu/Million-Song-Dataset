# src/evaluation.py

from typing import Dict, List, Set, Iterable, Any
from dataclasses import dataclass
import math


def precision_at_k(
    recs: List[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """
    Precision@K = (# of recommended items in ground_truth) / K
    """
    if k <= 0:
        return 0.0
    if not recs:
        return 0.0

    top_k = recs[:k]
    hits = sum(1 for x in top_k if x in ground_truth)
    return hits / float(k)


def recall_at_k(
    recs: List[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """
    Recall@K = (# of recommended items in ground_truth) / (# of ground truth items)
    """
    if not ground_truth:
        return 0.0

    top_k = recs[:k]
    hits = sum(1 for x in top_k if x in ground_truth)
    return hits / float(len(ground_truth))


def average_precision_at_k(
    recs: List[int],
    ground_truth: Set[int],
    k: int,
) -> float:
    """
    AP@K (Average Precision at K)

    AP@K = average of precision@i at each rank i where a hit occurs.

    If no hits, AP@K = 0.
    """
    if not ground_truth:
        return 0.0
    if not recs:
        return 0.0

    top_k = recs[:k]
    num_hits = 0
    sum_precisions = 0.0

    for i, item in enumerate(top_k, start=1):  # ranks are 1-based
        if item in ground_truth:
            num_hits += 1
            prec_i = num_hits / float(i)
            sum_precisions += prec_i

    if num_hits == 0:
        return 0.0

    return sum_precisions / float(len(ground_truth))


@dataclass
class EvalResult:
    model_name: str
    k: int
    avg_precision: float
    avg_recall: float
    map_k: float         # Mean Average Precision at K
    hit_rate: float      # fraction of users with at least one hit


def evaluate_model_at_k(
    model_name: str,
    recommendations: Dict[int, List[int]],
    test_songs: Dict[int, Set[int]],
    eval_users: Iterable[int],
    k: int,
) -> EvalResult:
    """
    Evaluate a single model's Top-K recommendations over a set of users.

    Args:
        model_name: name of the model (for logging)
        recommendations: dict user_idx -> list(song_idx) (ranked list)
        test_songs: dict user_idx -> set(song_idx) (ground truth)
        eval_users: list / iterable of user indices to evaluate
        k: the K in Precision@K / Recall@K / MAP@K

    Returns:
        EvalResult with averaged metrics.
    """
    precisions = []
    recalls = []
    ap_scores = []
    hits = 0
    total_users = 0

    for u in eval_users:
        gt = test_songs.get(u, set())
        recs = recommendations.get(u, [])

        # Skip users with empty ground truth just in case
        if not gt:
            continue

        total_users += 1

        p = precision_at_k(recs, gt, k)
        r = recall_at_k(recs, gt, k)
        ap = average_precision_at_k(recs, gt, k)

        precisions.append(p)
        recalls.append(r)
        ap_scores.append(ap)

        # hit if at least one correct item in top-k
        if any((x in gt) for x in recs[:k]):
            hits += 1

    if total_users == 0:
        # Avoid division by zero; return zeros
        return EvalResult(
            model_name=model_name,
            k=k,
            avg_precision=0.0,
            avg_recall=0.0,
            map_k=0.0,
            hit_rate=0.0,
        )

    avg_p = sum(precisions) / float(total_users)
    avg_r = sum(recalls) / float(total_users)
    map_k = sum(ap_scores) / float(total_users)
    hit_rate = hits / float(total_users)

    return EvalResult(
        model_name=model_name,
        k=k,
        avg_precision=avg_p,
        avg_recall=avg_r,
        map_k=map_k,
        hit_rate=hit_rate,
    )


def pretty_print_eval(result: EvalResult) -> None:
    """
    Nicely print evaluation results for a model.
    """
    print(f"\n=== Evaluation: {result.model_name} (K={result.k}) ===")
    print(f"  Avg Precision@{result.k}: {result.avg_precision:.4f}")
    print(f"  Avg Recall@{result.k}:    {result.avg_recall:.4f}")
    print(f"  MAP@{result.k}:           {result.map_k:.4f}")
    print(f"  Hit Rate:                 {result.hit_rate:.4f}")
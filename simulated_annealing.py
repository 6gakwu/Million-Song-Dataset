import random
import math
from typing import List, Dict, Iterable, Set


# Build simple surrogate scores from the Item-CF ranking:
# higher-ranked songs get higher weights.
def _build_rank_scores(ranked_songs: List[int]) -> Dict[int, float]:
    scores = {}
    for idx, s in enumerate(ranked_songs):
        scores[s] = 1.0 / float(idx + 1)
    return scores


# Objective function = sum of surrogate scores for the Top-K list
def _objective(top_k: List[int], score_dict: Dict[int, float]) -> float:
    return sum(score_dict.get(s, 0.0) for s in top_k)


# Simulated Annealing optimization on the Top-K portion of a ranked list
def _sa_optimize_topk(
    initial_list: List[int],
    score_dict: Dict[int, float],
    K: int,
    max_iters: int = 500,
    T_start: float = 1.0,
    T_end: float = 1e-3,
    cooling: float = 0.99,
) -> List[int]:

    n = len(initial_list)

    # Nothing to optimize if list too small
    if n <= 1:
        return initial_list

    # If user has fewer than K songs, shrink K
    if n < K:
        K = n

    # If still fewer than 2, no swaps possible
    if K <= 1:
        return initial_list

    # Initial state
    current_top = initial_list[:K]
    best_top = current_top[:]

    current_score = _objective(current_top, score_dict)
    best_score = current_score

    T = T_start
    iters = 0

    # Annealing loop
    while T > T_end and iters < max_iters:
        iters += 1

        # Propose a neighbor by swapping two random positions within Top-K
        i, j = random.sample(range(K), 2)

        new_top = current_top[:]
        new_top[i], new_top[j] = new_top[j], new_top[i]

        new_score = _objective(new_top, score_dict)
        delta = new_score - current_score

        # Accept if better, or with probability exp(delta / T) if worse
        if delta >= 0:
            current_top = new_top
            current_score = new_score
            if new_score > best_score:
                best_top = new_top
                best_score = new_score
        else:
            if random.random() < math.exp(delta / T):
                current_top = new_top
                current_score = new_score

        # Decrease temperature
        T *= cooling

    # Reconstruct full list: optimized Top-K + remaining items
    refined = best_top + initial_list[K:]
    return refined


# Apply SA refinement to all users
def recommend_sa_for_all_users(
    itemcf_recs: Dict[int, List[int]],
    train_songs: Dict[int, Set[int]],
    eval_users: Iterable[int],
    K: int,
    max_iters: int = 500,
    T_start: float = 1.0,
    T_end: float = 1e-3,
    cooling: float = 0.99,
) -> Dict[int, List[int]]:

    sa_recs: Dict[int, List[int]] = {}

    for u in eval_users:
        # Start from Item-CF recommended list
        base_list = itemcf_recs.get(u, [])

        # Remove items the user already knows
        base_list = [s for s in base_list if s not in train_songs.get(u, set())]

        # If no candidates, produce empty list
        if not base_list:
            sa_recs[u] = []
            continue

        # Build objective weights from ranking
        score_dict = _build_rank_scores(base_list)

        # Run SA optimization to refine Top-K ordering
        refined_list = _sa_optimize_topk(
            initial_list=base_list,
            score_dict=score_dict,
            K=K,
            max_iters=max_iters,
            T_start=T_start,
            T_end=T_end,
            cooling=cooling,
        )

        # Store final Top-K recommendations
        sa_recs[u] = refined_list[:K]

    return sa_recs

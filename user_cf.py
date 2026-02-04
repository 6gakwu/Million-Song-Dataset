# src/user_cf.py

from typing import Dict, Set, List, Tuple
from collections import defaultdict
import pandas as pd


def build_song_to_users_from_train(
    train_interactions: pd.DataFrame,
) -> Dict[int, Set[int]]:
    """
    Build a mapping song_idx -> set(user_idx) using ONLY train interactions.

    Args:
        train_interactions: DataFrame with ['user_idx', 'song_idx', 'play_count']

    Returns:
        song_to_users_train: dict mapping song_idx -> set of user_idx
    """
    song_to_users_train: Dict[int, Set[int]] = defaultdict(set)

    for _, row in train_interactions.iterrows():
        u = int(row["user_idx"])
        s = int(row["song_idx"])
        song_to_users_train[s].add(u)

    return song_to_users_train


def jaccard_similarity(
    items_u: Set[int],
    items_v: Set[int],
) -> float:
    """
    Compute Jaccard similarity between two sets of items.

    sim = |intersection| / |union|

    Args:
        items_u: set of song_idx for user u
        items_v: set of song_idx for user v

    Returns:
        similarity score in [0, 1]
    """
    if not items_u or not items_v:
        return 0.0

    inter = items_u.intersection(items_v)
    if not inter:
        return 0.0

    union = items_u.union(items_v)
    return len(inter) / len(union)


def find_user_neighbors(
    user_idx: int,
    train_songs: Dict[int, Set[int]],
    song_to_users_train: Dict[int, Set[int]],
    max_neighbors: int = 50,
    min_common_items: int = 2,
) -> List[Tuple[int, float]]:
    """
    Find similar users (neighbors) for a given user based on Jaccard similarity
    of their training song sets.

    Args:
        user_idx: target user index
        train_songs: dict user_idx -> set(song_idx)
        song_to_users_train: dict song_idx -> set(user_idx)
        max_neighbors: maximum neighbors to keep for this user
        min_common_items: minimum number of common songs to consider someone a neighbor

    Returns:
        neighbors: list of (neighbor_user_idx, similarity), sorted by similarity desc
    """
    items_u = train_songs.get(user_idx, set())
    if not items_u:
        return []

    # Candidate neighbors: users who share at least one song with u
    candidate_users: Set[int] = set()
    for s in items_u:
        candidate_users.update(song_to_users_train.get(s, set()))

    # Remove self from candidates
    if user_idx in candidate_users:
        candidate_users.remove(user_idx)

    neighbors: List[Tuple[int, float]] = []

    for v in candidate_users:
        items_v = train_songs.get(v, set())
        # quick prune: check if they share enough common items
        common = items_u.intersection(items_v)
        if len(common) < min_common_items:
            continue

        sim = jaccard_similarity(items_u, items_v)
        if sim <= 0.0:
            continue

        neighbors.append((v, sim))

    # Sort neighbors by similarity, keep top max_neighbors
    neighbors.sort(key=lambda x: x[1], reverse=True)
    if len(neighbors) > max_neighbors:
        neighbors = neighbors[:max_neighbors]

    return neighbors


def recommend_for_user_usercf(
    user_idx: int,
    train_songs: Dict[int, Set[int]],
    neighbors: List[Tuple[int, float]],
    K: int,
) -> List[int]:
    """
    Recommend Top-K songs for a single user using user-based CF.

    Scoring:
        For each candidate song s:
            score[s] = sum_{v in neighbors who listened to s} similarity(u, v)

    Args:
        user_idx: target user index
        train_songs: dict user_idx -> set(song_idx)
        neighbors: list of (neighbor_user_idx, similarity)
        K: number of recommendations

    Returns:
        recs: list of song_idx (up to length K)
    """
    known_items = train_songs.get(user_idx, set())
    if not neighbors:
        # Fall back: no neighbors, return empty list
        return []

    scores = defaultdict(float)

    for v, sim in neighbors:
        items_v = train_songs.get(v, set())
        for s in items_v:
            if s in known_items:
                continue  # don't recommend items the user already knows
            scores[s] += sim

    if not scores:
        return []

    # Sort candidate songs by score desc, use song_idx as tie-breaker
    ranked_songs = sorted(scores.keys(), key=lambda s: (scores[s], s), reverse=True)

    return ranked_songs[:K]


def recommend_user_cf_for_all_users(
    train_songs: Dict[int, Set[int]],
    train_interactions: pd.DataFrame,
    eval_users: List[int],
    K: int = 50,
    max_neighbors: int = 50,
    min_common_items: int = 2,
) -> Dict[int, List[int]]:
    """
    Run user-based collaborative filtering for all eval_users.

    Args:
        train_songs: dict user_idx -> set(song_idx) (training items)
        train_interactions: DataFrame of training interactions
        eval_users: list of user indices to generate recommendations for
        K: length of recommendation list
        max_neighbors: max number of neighbors per user
        min_common_items: minimum shared items to consider a neighbor

    Returns:
        recommendations: dict user_idx -> list(song_idx)
    """
    from collections import defaultdict as dd  # avoid conflict with outer name
    recommendations: Dict[int, List[int]] = {}

    # Build song -> users mapping from TRAIN data only
    song_to_users_train = build_song_to_users_from_train(train_interactions)

    for u in eval_users:
        neighs = find_user_neighbors(
            user_idx=u,
            train_songs=train_songs,
            song_to_users_train=song_to_users_train,
            max_neighbors=max_neighbors,
            min_common_items=min_common_items,
        )

        recs = recommend_for_user_usercf(
            user_idx=u,
            train_songs=train_songs,
            neighbors=neighs,
            K=K,
        )
        recommendations[u] = recs

    return recommendations

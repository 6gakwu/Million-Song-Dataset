# src/item_cf.py

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
        song_to_users_train: dict mapping song_idx -> set(user_idx)
    """
    song_to_users_train: Dict[int, Set[int]] = defaultdict(set)

    for _, row in train_interactions.iterrows():
        u = int(row["user_idx"])
        s = int(row["song_idx"])
        song_to_users_train[s].add(u)

    return song_to_users_train


def jaccard_similarity_usersets(
    users_i: Set[int],
    users_j: Set[int],
) -> float:
    """
    Jaccard similarity between two sets of users:
        sim(i, j) = |U_i ∩ U_j| / |U_i ∪ U_j|
    """
    if not users_i or not users_j:
        return 0.0

    inter = users_i.intersection(users_j)
    if not inter:
        return 0.0

    union = users_i.union(users_j)
    return len(inter) / len(union)


def recommend_for_user_itemcf(
    user_idx: int,
    train_songs: Dict[int, Set[int]],
    song_to_users_train: Dict[int, Set[int]],
    K: int = 50,
    min_common_users: int = 2,
) -> List[int]:
    """
    Recommend Top-K songs for a single user using item-based CF.

    Steps:
      1. Get songs the user already knows (train_songs[user_idx])
      2. For each known song i:
           - Look at users who listened to i (song_to_users_train[i])
           - Collect songs j that those users listened to
      3. For each candidate song j (that user has not seen):
           - Compute item-item similarity sim(i, j) for all i in user's known songs
           - score[j] = sum_of_sim(i, j) over i in known songs
      4. Rank candidates by score and return top-K.

    Args:
        user_idx: target user index
        train_songs: dict user_idx -> set(song_idx)
        song_to_users_train: dict song_idx -> set(user_idx)
        K: number of recommendations to return
        min_common_users: minimum shared users for (i, j) to be considered similar

    Returns:
        recs: list of song_idx (up to length K)
    """
    known_items = train_songs.get(user_idx, set())
    if not known_items:
        return []

    # Step 1: build candidate set = songs co-listened with user's items
    candidate_songs: Set[int] = set()

    # Users who listened to user's songs
    for i in known_items:
        users_who_listened_i = song_to_users_train.get(i, set())
        for u2 in users_who_listened_i:
            # Add all songs from these users
            candidate_songs.update(train_songs.get(u2, set()))

    # Remove songs the user already knows
    candidate_songs.difference_update(known_items)

    if not candidate_songs:
        return []

    scores = defaultdict(float)

    # Step 2: score each candidate song j by similarity to items in user's profile
    for j in candidate_songs:
        users_j = song_to_users_train.get(j, set())
        if not users_j:
            continue

        total_sim = 0.0
        for i in known_items:
            users_i = song_to_users_train.get(i, set())
            if not users_i:
                continue

            # quick prune: check if they share enough common users
            if len(users_i.intersection(users_j)) < min_common_users:
                continue

            sim_ij = jaccard_similarity_usersets(users_i, users_j)
            if sim_ij <= 0.0:
                continue

            total_sim += sim_ij

        if total_sim > 0.0:
            scores[j] = total_sim

    if not scores:
        return []

    # Step 3: rank candidates by score
    ranked_songs = sorted(scores.keys(), key=lambda s: (scores[s], s), reverse=True)

    return ranked_songs[:K]


def recommend_item_cf_for_all_users(
    train_songs: Dict[int, Set[int]],
    train_interactions: pd.DataFrame,
    eval_users: List[int],
    K: int = 50,
    min_common_users: int = 2,
) -> Dict[int, List[int]]:
    """
    Run item-based collaborative filtering for all eval_users.

    Args:
        train_songs: dict user_idx -> set(song_idx) in training
        train_interactions: DataFrame with training interactions
        eval_users: list of user indices to generate recommendations for
        K: length of recommendation list
        min_common_users: minimum number of shared users for two items

    Returns:
        recommendations: dict user_idx -> list(song_idx)
    """
    recommendations: Dict[int, List[int]] = {}

    # Precompute song -> users mapping from TRAIN data
    song_to_users_train = build_song_to_users_from_train(train_interactions)

    for u in eval_users:
        recs = recommend_for_user_itemcf(
            user_idx=u,
            train_songs=train_songs,
            song_to_users_train=song_to_users_train,
            K=K,
            min_common_users=min_common_users,
        )
        recommendations[u] = recs

    return recommendations

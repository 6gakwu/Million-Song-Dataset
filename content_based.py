# src/content_based.py

from typing import Dict, Set, List
import numpy as np
import pandas as pd


def build_item_features_from_interactions(
    train_interactions: pd.DataFrame,
) -> Dict[int, np.ndarray]:
    """
    Build simple numeric feature vectors for each song based on training interactions.

    For each song, we compute:
      - total_play: total play_count across all users
      - user_count: number of unique users who played the song
      - avg_play_per_user: total_play / user_count

    We then apply log1p to the first two features to reduce skew and store
    feature vectors as small numpy arrays.

    Args:
        train_interactions: DataFrame with ['user_idx', 'song_idx', 'play_count']

    Returns:
        item_features: dict mapping song_idx -> np.ndarray of shape (3,)
    """
    # Aggregate statistics per song
    stats = (
        train_interactions.groupby("song_idx")
        .agg(
            total_play=("play_count", "sum"),
            user_count=("user_idx", "nunique"),
        )
        .reset_index()
    )

    # Avoid division by zero
    stats["avg_play_per_user"] = stats["total_play"] / stats["user_count"].replace(0, np.nan)

    item_features: Dict[int, np.ndarray] = {}

    for _, row in stats.iterrows():
        s = int(row["song_idx"])
        total_play = float(row["total_play"])
        user_count = float(row["user_count"])
        avg_play = float(row["avg_play_per_user"])

        # Log-transform the first two features to reduce skew
        f1 = np.log1p(total_play)
        f2 = np.log1p(user_count)
        f3 = avg_play if not np.isnan(avg_play) else 0.0

        item_features[s] = np.array([f1, f2, f3], dtype=np.float32)

    return item_features


def normalize_feature_vectors(
    item_features: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    Normalize each feature vector to unit length (L2 norm = 1).
    This allows us to use simple dot products as cosine similarity.

    Args:
        item_features: dict song_idx -> np.ndarray

    Returns:
        normalized_features: dict song_idx -> np.ndarray (unit norm)
    """
    normalized_features: Dict[int, np.ndarray] = {}

    for s, vec in item_features.items():
        norm = np.linalg.norm(vec)
        if norm > 0:
            normalized_features[s] = vec / norm
        else:
            # If zero vector, keep as zero
            normalized_features[s] = vec

    return normalized_features


def build_user_profiles(
    train_songs: Dict[int, Set[int]],
    item_features: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    Build a profile vector for each user by averaging the feature vectors
    of the songs in their training set.

    Args:
        train_songs: dict user_idx -> set(song_idx) in training
        item_features: dict song_idx -> feature vector (unit norm recommended)

    Returns:
        user_profiles: dict user_idx -> np.ndarray (user profile vector)
    """
    user_profiles: Dict[int, np.ndarray] = {}

    for u, songs in train_songs.items():
        vecs = [item_features[s] for s in songs if s in item_features]
        if not vecs:
            continue

        mat = np.stack(vecs, axis=0)  # shape: (num_songs, feature_dim)
        profile = mat.mean(axis=0)

        # Normalize user profile too, so dot(product) ≈ cosine similarity
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm

        user_profiles[u] = profile

    return user_profiles


def recommend_for_user_content(
    user_idx: int,
    train_songs: Dict[int, Set[int]],
    user_profiles: Dict[int, np.ndarray],
    item_features: Dict[int, np.ndarray],
    K: int = 50,
) -> List[int]:
    """
    Recommend Top-K songs for a single user using a content-based approach.

    Steps:
      1. Get the user's profile vector.
      2. For all songs with feature vectors:
           - Skip songs already in user's training set
           - Compute similarity = dot(user_profile, song_feature)
      3. Rank songs by similarity and return top-K.

    Args:
        user_idx: target user index
        train_songs: dict user_idx -> set(song_idx)
        user_profiles: dict user_idx -> profile vector
        item_features: dict song_idx -> feature vector (unit norm)
        K: number of recommendations to return

    Returns:
        recs: list of song_idx (up to length K)
    """
    if user_idx not in user_profiles:
        return []

    profile = user_profiles[user_idx]
    known = train_songs.get(user_idx, set())

    scores = {}
    for s, vec in item_features.items():
        if s in known:
            continue
        # dot product between unit vectors = cosine similarity
        sim = float(np.dot(profile, vec))
        if sim > 0.0:
            scores[s] = sim

    if not scores:
        return []

    ranked_songs = sorted(scores.keys(), key=lambda s: (scores[s], s), reverse=True)
    return ranked_songs[:K]


def recommend_content_for_all_users(
    train_songs: Dict[int, Set[int]],
    train_interactions: pd.DataFrame,
    eval_users: List[int],
    K: int = 50,
) -> Dict[int, List[int]]:
    """
    Run content-based recommendation for all eval_users.

    Args:
        train_songs: dict user_idx -> set(song_idx)
        train_interactions: DataFrame with training interactions
        eval_users: list of user indices to generate recommendations for
        K: length of recommendation list

    Returns:
        recommendations: dict user_idx -> list(song_idx)
    """
    # 1. Build item feature vectors from interactions
    raw_item_features = build_item_features_from_interactions(train_interactions)
    item_features = normalize_feature_vectors(raw_item_features)

    # 2. Build user profiles
    user_profiles = build_user_profiles(train_songs, item_features)

    recommendations: Dict[int, List[int]] = {}

    for u in eval_users:
        recs = recommend_for_user_content(
            user_idx=u,
            train_songs=train_songs,
            user_profiles=user_profiles,
            item_features=item_features,
            K=K,
        )
        recommendations[u] = recs

    return recommendations

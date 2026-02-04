# src/popularity.py

from typing import Dict, List, Set
import pandas as pd


def compute_popularity(
    train_interactions: pd.DataFrame,
) -> Dict[int, int]:
    """
    Compute global song popularity based on the training interactions.

    Args:
        train_interactions: DataFrame with columns ['user_idx', 'song_idx', 'play_count']

    Returns:
        popularity: dict mapping song_idx -> total play_count across all users
    """
    # Group by song and sum play counts
    grouped = train_interactions.groupby("song_idx")["play_count"].sum()

    # Convert to plain dict: {song_idx: total_count}
    popularity: Dict[int, int] = grouped.to_dict()
    return popularity


def get_songs_ranked_by_popularity(
    popularity: Dict[int, int]
) -> List[int]:
    """
    Return a list of song indices sorted by decreasing popularity.

    Args:
        popularity: dict song_idx -> total play_count

    Returns:
        songs_ranked: list of song_idx from most to least popular
    """
    # Sort songs by count (descending). Tie-breaker is song_idx (for stability).
    songs_ranked = sorted(
        popularity.keys(),
        key=lambda s: (popularity[s], s),
        reverse=True,
    )
    return songs_ranked


def recommend_for_user_popularity(
    user_idx: int,
    train_songs: Dict[int, Set[int]],
    songs_ranked: List[int],
    K: int,
) -> List[int]:
    """
    Recommend Top-K songs for a single user using global popularity,
    skipping songs already in the user's training set.

    Args:
        user_idx: integer user index
        train_songs: dict user_idx -> set(song_idx) (training items)
        songs_ranked: global list of songs sorted by popularity
        K: number of recommendations to generate

    Returns:
        recs: list of song_idx (length up to K)
    """
    known = train_songs.get(user_idx, set())
    recs: List[int] = []

    for s in songs_ranked:
        if s in known:
            continue
        recs.append(s)
        if len(recs) >= K:
            break

    return recs


def recommend_popularity_for_all_users(
    train_songs: Dict[int, Set[int]],
    songs_ranked: List[int],
    eval_users: List[int],
    K: int,
) -> Dict[int, List[int]]:
    """
    Recommend Top-K songs for all users in eval_users using global popularity.

    Args:
        train_songs: dict user_idx -> set(song_idx) in training
        songs_ranked: global list of songs sorted by popularity
        eval_users: list of user indices we want to evaluate
        K: length of the recommendation list for each user

    Returns:
        recommendations: dict user_idx -> list(song_idx) of length K (or less if not enough songs)
    """
    recommendations: Dict[int, List[int]] = {}

    for u in eval_users:
        recs = recommend_for_user_popularity(
            user_idx=u,
            train_songs=train_songs,
            songs_ranked=songs_ranked,
            K=K,
        )
        recommendations[u] = recs

    return recommendations

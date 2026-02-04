# src/splitter.py

import random
from typing import Dict, Set, Tuple, List
import pandas as pd


def train_test_split_user_songs(
    user_to_songs: Dict[int, Set[int]],
    train_fraction: float = 0.8,
    min_items_per_user: int = 2,
    seed: int = 42,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]], List[int]]:
    """
    Create a per-user train/test split based on the set of songs they listened to.

    Args:
        user_to_songs: dictionary mapping user_idx -> set(song_idx)
        train_fraction: fraction of items to keep in train for each user (e.g., 0.8)
        min_items_per_user: users with fewer than this number of items
                            will be kept in train only (empty test)
        seed: random seed for reproducibility

    Returns:
        train_songs: dict user_idx -> set of song_idx in train
        test_songs: dict user_idx -> set of song_idx in test
        eval_users: list of users that have a non-empty test set
                    (these are the users we can evaluate on)
    """
    random.seed(seed)

    train_songs: Dict[int, Set[int]] = {}
    test_songs: Dict[int, Set[int]] = {}
    eval_users: List[int] = []

    for u, songs_set in user_to_songs.items():
        songs = list(songs_set)
        if len(songs) < min_items_per_user:
            # Not enough items to split; put everything in train, empty test
            train_songs[u] = set(songs)
            test_songs[u] = set()
            continue

        random.shuffle(songs)

        cut = int(len(songs) * train_fraction)
        if cut <= 0:
            # Edge case: if train_fraction is too small
            cut = 1
        if cut >= len(songs):
            # Edge case: if train_fraction is too large
            cut = len(songs) - 1

        train_set = set(songs[:cut])
        test_set = set(songs[cut:])

        train_songs[u] = train_set
        test_songs[u] = test_set

        if len(test_set) > 0:
            eval_users.append(u)

    return train_songs, test_songs, eval_users


def split_interactions_by_train_test(
    interactions: pd.DataFrame,
    train_songs: Dict[int, Set[int]],
    test_songs: Dict[int, Set[int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the interactions DataFrame into train and test based on the
    per-user train_songs / test_songs sets.

    Args:
        interactions: DataFrame with columns ['user_idx', 'song_idx', 'play_count']
        train_songs: dict user_idx -> set(song_idx) for train
        test_songs: dict user_idx -> set(song_idx) for test

    Returns:
        train_interactions: subset of interactions belonging to train
        test_interactions: subset of interactions belonging to test
    """
    # For speed, we'll define small helper functions that check membership.
    def is_train_row(row):
        u = row["user_idx"]
        s = row["song_idx"]
        return s in train_songs.get(u, set())

    def is_test_row(row):
        u = row["user_idx"]
        s = row["song_idx"]
        return s in test_songs.get(u, set())

    train_mask = interactions.apply(is_train_row, axis=1)
    test_mask = interactions.apply(is_test_row, axis=1)

    train_interactions = interactions[train_mask].reset_index(drop=True)
    test_interactions = interactions[test_mask].reset_index(drop=True)

    return train_interactions, test_interactions
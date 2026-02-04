# src/data_loader.py

import os
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd


def load_canonical_users(users_path: str):
    """
    Load kaggle_users.txt and return:
      - user_ids: list of user_id strings in canonical order
      - user_id_to_index: dict mapping user_id to integer index (0..n-1)
    """
    with open(users_path, "r") as f:
        user_ids = [line.strip() for line in f if line.strip()]

    user_id_to_index = {uid: i for i, uid in enumerate(user_ids)}
    return user_ids, user_id_to_index


def load_canonical_songs(songs_path: str):
    """
    Load kaggle_songs.txt and return:
      - song_ids: list where index = canonical song index, value = song_id
      - song_id_to_index: dict mapping song_id to string/int index (as in file)

    In the original MSD script, the indices are stored as strings.
    For convenience we convert them to int here.
    """
    song_ids = []
    song_id_to_index: Dict[str, int] = {}

    with open(songs_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            song_id, idx_str = parts
            idx = int(idx_str)
            # ensure list is large enough
            if idx >= len(song_ids):
                song_ids.extend([None] * (idx - len(song_ids) + 1))
            song_ids[idx] = song_id
            song_id_to_index[song_id] = idx

    return song_ids, song_id_to_index


def load_triplets(
    triplets_path: str,
    user_id_to_index: Dict[str, int],
    song_id_to_index: Dict[str, int],
    max_rows: int = None,
):
    """
    Load kaggle_visible_evaluation_triplets.txt and convert:
      - user_id -> user_index
      - song_id -> song_index

    Returns:
      - interactions: pd.DataFrame with columns [user_idx, song_idx, play_count]
      - user_to_songs: Dict[int, Set[int]] mapping user_idx -> set of song_idx
      - song_to_users: Dict[int, Set[int]] mapping song_idx -> set of user_idx

    """
    user_indices: List[int] = []
    song_indices: List[int] = []
    play_counts: List[int] = []

    user_to_songs: Dict[int, Set[int]] = {}
    song_to_users: Dict[int, Set[int]] = {}

    with open(triplets_path, "r") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break

            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            user_id, song_id, count_str = parts
            if user_id not in user_id_to_index:
                # All test users should be in kaggle_users.txt
                continue
            if song_id not in song_id_to_index:
                # All songs should be in kaggle_songs.txt
                continue

            u_idx = user_id_to_index[user_id]
            s_idx = song_id_to_index[song_id]
            try:
                c = int(count_str)
            except ValueError:
                c = 1  # fallback

            user_indices.append(u_idx)
            song_indices.append(s_idx)
            play_counts.append(c)

            # build user_to_songs
            if u_idx not in user_to_songs:
                user_to_songs[u_idx] = set()
            user_to_songs[u_idx].add(s_idx)

            # build song_to_users
            if s_idx not in song_to_users:
                song_to_users[s_idx] = set()
            song_to_users[s_idx].add(u_idx)

    interactions = pd.DataFrame(
        {
            "user_idx": user_indices,
            "song_idx": song_indices,
            "play_count": play_counts,
        }
    )

    return interactions, user_to_songs, song_to_users


def load_all_data(
    data_dir: str,
    max_triplets: int = None,
):
    """
    Convenience function to load everything at once.

    Args:
        data_dir: directory containing the three data files
        max_triplets: optional cap on number of triplets to read

    Returns:
        user_ids, user_id_to_index
        song_ids, song_id_to_index
        interactions, user_to_songs, song_to_users
    """
    users_path = os.path.join(data_dir, "kaggle_users.txt")
    songs_path = os.path.join(data_dir, "kaggle_songs.txt")
    triplets_path = os.path.join(data_dir, "kaggle_visible_evaluation_triplets.txt")

    user_ids, user_id_to_index = load_canonical_users(users_path)
    song_ids, song_id_to_index = load_canonical_songs(songs_path)

    interactions, user_to_songs, song_to_users = load_triplets(
        triplets_path,
        user_id_to_index,
        song_id_to_index,
        max_rows=max_triplets,
    )

    return (
        user_ids,
        user_id_to_index,
        song_ids,
        song_id_to_index,
        interactions,
        user_to_songs,
        song_to_users,
    )

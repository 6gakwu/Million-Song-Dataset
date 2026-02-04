from src.data_loader import load_all_data
from src.splitter import train_test_split_user_songs, split_interactions_by_train_test

from src.popularity import (
    compute_popularity,
    get_songs_ranked_by_popularity,
    recommend_popularity_for_all_users,
)

from src.user_cf import recommend_user_cf_for_all_users
from src.item_cf import recommend_item_cf_for_all_users
from src.content_based import recommend_content_for_all_users
from src.simulated_annealing import recommend_sa_for_all_users

from src.evaluation import evaluate_model_at_k, pretty_print_eval

import pandas as pd
import matplotlib.pyplot as plt
import time


def main():
    data_dir = "data"

    # Load user IDs, song IDs, and all interaction triplets
    (
        user_ids,
        user_id_to_index,
        song_ids,
        song_id_to_index,
        interactions,
        user_to_songs,
        song_to_users,
    ) = load_all_data(data_dir, max_triplets=None)

    print("Number of users:", len(user_ids))
    print("Number of songs:", len(song_ids))
    print("Interactions loaded:", len(interactions))

    # Split each user's history into train/test sets (80/20)
    train_songs, test_songs, eval_users = train_test_split_user_songs(
        user_to_songs,
        train_fraction=0.8,
        min_items_per_user=2,
        seed=42,
    )

    print("Users with data:", len(user_to_songs))
    print("Eval users:", len(eval_users))

    # Build train/test interaction tables
    train_interactions, test_interactions = split_interactions_by_train_test(
        interactions,
        train_songs,
        test_songs,
    )

    print("Train interactions:", len(train_interactions))
    print("Test interactions:", len(test_interactions))

    # Use a reduced set of users for faster experimentation
    small_eval_users = eval_users[:500]
    print(f"Using {len(small_eval_users)} eval users for fast testing")

    K = 50
    sample_user = small_eval_users[0]

    # ------------------------------
    # Popularity baseline model
    # ------------------------------
    print("\n=== Popularity Baseline ===")
    popularity = compute_popularity(train_interactions)
    songs_ranked = get_songs_ranked_by_popularity(popularity)

    start = time.time()
    popularity_recs = recommend_popularity_for_all_users(
        train_songs=train_songs,
        songs_ranked=songs_ranked,
        eval_users=small_eval_users,
        K=K,
    )
    pop_time = time.time() - start
    print(f"Popularity recommendation time: {pop_time:.2f} seconds")

    print(f"\n[Popularity] Sample user {sample_user}")
    print("  Train:", list(train_songs[sample_user])[:10])
    print("  Test:", list(test_songs[sample_user]))
    print("  Recs:", popularity_recs[sample_user][:10])

    # ------------------------------
    # User-Based Collaborative Filtering
    # ------------------------------
    print("\n=== User-Based CF ===")
    start = time.time()
    usercf_recs = recommend_user_cf_for_all_users(
        train_songs=train_songs,
        train_interactions=train_interactions,
        eval_users=small_eval_users,
        K=K,
        max_neighbors=50,
        min_common_items=2,
    )
    usercf_time = time.time() - start
    print(f"User-CF recommendation time: {usercf_time:.2f} seconds")

    print(f"\n[User-CF] Sample user {sample_user}")
    print("  Train:", list(train_songs[sample_user])[:10])
    print("  Test:", list(test_songs[sample_user]))
    print("  Recs:", usercf_recs[sample_user][:10])

    # ------------------------------
    # Item-Based Collaborative Filtering
    # ------------------------------
    print("\n=== Item-Based CF ===")
    start = time.time()
    itemcf_recs = recommend_item_cf_for_all_users(
        train_songs=train_songs,
        train_interactions=train_interactions,
        eval_users=small_eval_users,
        K=K,
        min_common_users=2,
    )
    itemcf_time = time.time() - start
    print(f"Item-CF recommendation time: {itemcf_time:.2f} seconds")

    print(f"\n[Item-CF] Sample user {sample_user}")
    print("  Train:", list(train_songs[sample_user])[:10])
    print("  Test:", list(test_songs[sample_user]))
    print("  Recs:", itemcf_recs[sample_user][:10])

    # ------------------------------
    # Content-Based Filtering
    # ------------------------------
    print("\n=== Content-Based Recommendation ===")
    start = time.time()
    content_recs = recommend_content_for_all_users(
        train_songs=train_songs,
        train_interactions=train_interactions,
        eval_users=small_eval_users,
        K=K,
    )
    content_time = time.time() - start
    print(f"Content-Based recommendation time: {content_time:.2f} seconds")

    print(f"\n[Content-Based] Sample user {sample_user}")
    print("  Train:", list(train_songs[sample_user])[:10])
    print("  Test:", list(test_songs[sample_user]))
    print("  Recs:", content_recs[sample_user][:10])

    # ------------------------------
    # Simulated Annealing Re-Ranking (applied to Item-CF results)
    # ------------------------------
    print("\n=== Simulated Annealing Re-Ranking (on Item-CF) ===")
    start = time.time()
    sa_recs = recommend_sa_for_all_users(
        itemcf_recs=itemcf_recs,
        train_songs=train_songs,
        eval_users=small_eval_users,
        K=K,
        max_iters=500,
        T_start=1.0,
        T_end=1e-3,
        cooling=0.99,
    )
    sa_time = time.time() - start
    print(f"Simulated Annealing recommendation time: {sa_time:.2f} seconds")

    print(f"\n[SA] Sample user {sample_user}")
    print("  Train:", list(train_songs[sample_user])[:10])
    print("  Test:", list(test_songs[sample_user]))
    print("  Recs:", sa_recs[sample_user][:10])

    # ------------------------------
    # Evaluate all models using Top-K metrics
    # ------------------------------
    print("\n=== MODEL EVALUATION @K ===")

    pop_result = evaluate_model_at_k("Popularity", popularity_recs, test_songs, small_eval_users, K)
    usercf_result = evaluate_model_at_k("User-CF", usercf_recs, test_songs, small_eval_users, K)
    itemcf_result = evaluate_model_at_k("Item-CF", itemcf_recs, test_songs, small_eval_users, K)
    content_result = evaluate_model_at_k("Content-Based", content_recs, test_songs, small_eval_users, K)
    sa_result = evaluate_model_at_k("Simulated Annealing", sa_recs, test_songs, small_eval_users, K)

    pretty_print_eval(pop_result)
    pretty_print_eval(usercf_result)
    pretty_print_eval(itemcf_result)
    pretty_print_eval(content_result)
    pretty_print_eval(sa_result)

    # ------------------------------
    # Combine results into comparison table
    # ------------------------------
    print("\n=== MODEL COMPARISON TABLE ===")

    results_df = pd.DataFrame([
        {
            "Model": pop_result.model_name,
            "Precision@K": pop_result.avg_precision,
            "Recall@K": pop_result.avg_recall,
            "MAP@K": pop_result.map_k,
            "HitRate": pop_result.hit_rate,
            "Time(s)": pop_time,
        },
        {
            "Model": usercf_result.model_name,
            "Precision@K": usercf_result.avg_precision,
            "Recall@K": usercf_result.avg_recall,
            "MAP@K": usercf_result.map_k,
            "HitRate": usercf_result.hit_rate,
            "Time(s)": usercf_time,
        },
        {
            "Model": itemcf_result.model_name,
            "Precision@K": itemcf_result.avg_precision,
            "Recall@K": itemcf_result.avg_recall,
            "MAP@K": itemcf_result.map_k,
            "HitRate": itemcf_result.hit_rate,
            "Time(s)": itemcf_time,
        },
        {
            "Model": content_result.model_name,
            "Precision@K": content_result.avg_precision,
            "Recall@K": content_result.avg_recall,
            "MAP@K": content_result.map_k,
            "HitRate": content_result.hit_rate,
            "Time(s)": content_time,
        },
        {
            "Model": sa_result.model_name,
            "Precision@K": sa_result.avg_precision,
            "Recall@K": sa_result.avg_recall,
            "MAP@K": sa_result.map_k,
            "HitRate": sa_result.hit_rate,
            "Time(s)": sa_time,
        },
    ])

    print(results_df.to_string(index=False))

     # ------------------------------
    # Runtime analysis graph
    # ------------------------------
    print("\n=== RUNTIME SUMMARY (seconds) ===")
    print(f"  Popularity        : {pop_time:.2f} s")
    print(f"  User-CF           : {usercf_time:.2f} s")
    print(f"  Item-CF           : {itemcf_time:.2f} s")
    print(f"  Content-Based     : {content_time:.2f} s")
    print(f"  Simulated Annealing: {sa_time:.2f} s")

    # Plot metric comparison
    metrics = ["Precision@K", "Recall@K", "MAP@K", "HitRate"]

    for metric in metrics:
        plt.figure(figsize=(7, 5))
        plt.bar(results_df["Model"], results_df[metric])
        plt.title(f"{metric} Comparison Across Models (K={K})")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


    print("\n=== END OF EXECUTION ===")


if __name__ == "__main__":
    main()

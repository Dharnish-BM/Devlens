"""
DevLens Phase 5: K-Means Clustering & Artifact Generation.

- Loads current feature vectors via repository.get_all_current_features()
- Preprocesses & scales features using feature_engineering.prepare_scaled_feature_matrix()
- Evaluates k=2..10 with Silhouette score & Elbow method (Inertia)
- Saves k_selection.png, kmeans_model.joblib, and scaler.joblib to models/artifacts/
- Saves cluster assignments & distance to centroid to the database
- Summarizes clusters by size and top 5 distinguishing features
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from devlens.db.session import get_session
from devlens.db.repository import (
    get_all_current_features,
    get_developer,
    get_latest_snapshot,
    upsert_cluster_assignment,
)
from devlens.features.feature_engineering import prepare_scaled_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIRS = [
    Path("models/artifacts"),
    Path("devlens/models/artifacts"),
]


def ensure_artifact_dirs():
    """Ensure artifact directories exist."""
    for d in ARTIFACT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def save_artifact(obj: Any, filename: str):
    """Save an object or plot to all artifact directory paths."""
    ensure_artifact_dirs()
    for d in ARTIFACT_DIRS:
        path = d / filename
        if isinstance(obj, plt.Figure):
            obj.savefig(path, dpi=300, bbox_inches="tight")
        else:
            joblib.dump(obj, path)
        logger.info(f"Saved artifact: {path}")


def evaluate_k_selection(
    scaled_df: pd.DataFrame,
    k_range: range = range(2, 11)
) -> Tuple[Dict[int, float], Dict[int, float], int, plt.Figure]:
    """Run K-Means across k_range, computing inertia & silhouette scores.
    
    Returns:
        (inertias, silhouette_scores, best_k_by_silhouette, figure)
    """
    inertias: Dict[int, float] = {}
    silhouettes: Dict[int, float] = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_df)
        inertias[k] = float(kmeans.inertia_)
        silhouettes[k] = float(silhouette_score(scaled_df, labels))

    best_k = max(silhouettes, key=silhouettes.get)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Elbow (Inertia)
    ax1.plot(list(k_range), [inertias[k] for k in k_range], 'o-', color='#2b5c8f', linewidth=2, markersize=8)
    ax1.set_title("Elbow Method (Inertia vs k)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax1.set_ylabel("Inertia (Sum of Squared Distances)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Silhouette Score
    ax2.plot(list(k_range), [silhouettes[k] for k in k_range], 's-', color='#d95f02', linewidth=2, markersize=8)
    ax2.axvline(x=best_k, color='#2ca02c', linestyle='--', label=f"Max Silhouette (k={best_k})")
    ax2.set_title("Silhouette Score vs k", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax2.set_ylabel("Silhouette Score", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    return inertias, silhouettes, best_k, fig


def run_clustering(
    n_clusters: Optional[int] = None,
    save_db: bool = True
) -> Dict[str, Any]:
    """Execute full Phase 5 clustering pipeline.
    
    1. Loads features from DB.
    2. Scales & preprocesses features.
    3. Evaluates k=2..10 and generates plot artifact.
    4. Fits K-Means with chosen k (or highest silhouette k if None).
    5. Saves model + scaler artifacts.
    6. Persists cluster assignments to DB.
    7. Computes and logs cluster summary statistics.
    """
    logger.info("Loading feature vectors from database...")
    with get_session() as session:
        unscaled_df = get_all_current_features(session)

    if unscaled_df.empty:
        raise ValueError("No feature data found in database. Run feature engineering first.")

    logger.info(f"Loaded {len(unscaled_df)} developer feature vectors.")

    # 1. Scale feature matrix
    scaled_df, scaler, dropped_cols = prepare_scaled_feature_matrix(unscaled_df, drop_zero_variance=True)
    logger.info(f"Scaled feature matrix shape: {scaled_df.shape}")

    # 2. Evaluate K selection
    inertias, silhouettes, auto_best_k, k_fig = evaluate_k_selection(scaled_df)
    save_artifact(k_fig, "k_selection.png")
    plt.close(k_fig)

    chosen_k = n_clusters if n_clusters is not None else auto_best_k
    logger.info(f"Selected k={chosen_k} for final K-Means model (Auto-best silhouette k={auto_best_k}).")

    # 3. Fit final K-Means model
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_df)
    scaled_df["cluster"] = cluster_labels
    unscaled_df["cluster"] = cluster_labels

    # Distance to assigned centroid for each developer
    centroids = kmeans.cluster_centers_
    distances = []
    feature_matrix_np = scaled_df.drop(columns=["cluster"]).values
    for idx, cluster_id in enumerate(cluster_labels):
        dist = np.linalg.norm(feature_matrix_np[idx] - centroids[cluster_id])
        distances.append(float(dist))
    scaled_df["distance_to_centroid"] = distances

    # 4. Save model artifacts
    save_artifact(kmeans, "kmeans_model.joblib")
    save_artifact(scaler, "scaler.joblib")

    # 5. Save to database
    if save_db:
        logger.info("Persisting cluster assignments to database...")
        with get_session() as session:
            for username, row in scaled_df.iterrows():
                dev = get_developer(session, str(username))
                if dev:
                    snapshot = get_latest_snapshot(session, dev.id)
                    if snapshot:
                        upsert_cluster_assignment(
                            session=session,
                            snapshot_id=snapshot.id,
                            cluster_id=int(row["cluster"]),
                            distance_to_centroid=float(row["distance_to_centroid"]),
                        )
            session.commit()
        logger.info("Cluster assignments successfully stored in DB.")

    # 6. Generate Cluster Summaries
    cluster_summaries = []
    feature_cols = [c for c in scaled_df.columns if c not in ("cluster", "distance_to_centroid")]
    global_scaled_means = scaled_df[feature_cols].mean()

    for c_id in range(chosen_k):
        c_scaled = scaled_df[scaled_df["cluster"] == c_id][feature_cols]
        c_unscaled = unscaled_df[unscaled_df["cluster"] == c_id].drop(columns=["cluster"])
        c_size = len(c_scaled)

        # Distinguishing features by mean deviation from global mean in scaled space
        scaled_diff = c_scaled.mean() - global_scaled_means
        top_diffs = scaled_diff.abs().sort_values(ascending=False).head(5)

        top_features = []
        for feat in top_diffs.index:
            dev_val = scaled_diff[feat]
            unscaled_val = c_unscaled[feat].mean() if feat in c_unscaled.columns else 0.0
            top_features.append({
                "feature": feat,
                "z_deviation": float(dev_val),
                "unscaled_cluster_mean": float(unscaled_val),
            })

        cluster_summaries.append({
            "cluster_id": c_id,
            "size": c_size,
            "pct_of_total": float(c_size / len(scaled_df) * 100),
            "top_features": top_features,
        })

    return {
        "chosen_k": chosen_k,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "scaled_df": scaled_df,
        "unscaled_df": unscaled_df,
        "kmeans": kmeans,
        "scaler": scaler,
        "cluster_summaries": cluster_summaries,
    }


def print_cluster_summary_report(results: Dict[str, Any]) -> None:
    """Print readable per-cluster summary report."""
    print("\n" + "=" * 80)
    print(f"  DEV LENS - PHASE 5 K-MEANS CLUSTERING SUMMARY (k={results['chosen_k']})")
    print("=" * 80)
    
    print("\n--- Model Performance Across k=2..10 ---")
    print(f"{'k':<5} {'Inertia':<15} {'Silhouette Score':<20}")
    print("-" * 40)
    for k in sorted(results["inertias"].keys()):
        marker = " (Selected)" if k == results["chosen_k"] else ""
        print(f"{k:<5} {results['inertias'][k]:<15.4f} {results['silhouettes'][k]:<20.4f}{marker}")

    print("\n--- Per-Cluster Composition & Distinguishing Features ---")
    for summary in results["cluster_summaries"]:
        c_id = summary["cluster_id"]
        size = summary["size"]
        pct = summary["pct_of_total"]
        print(f"\n[Cluster {c_id}] Size: {size} developers ({pct:.1f}% of cohort)")
        print(f"Top 5 Distinguishing Features (by Z-score deviation from global mean):")
        for f_info in summary["top_features"]:
            feat = f_info["feature"]
            z_dev = f_info["z_deviation"]
            unscaled_m = f_info["unscaled_cluster_mean"]
            direction = "ABOVE" if z_dev > 0 else "BELOW"
            print(f"  • {feat:<32}: {z_dev:+6.2f} std ({direction} avg) | Raw Mean: {unscaled_m:.4f}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    results = run_clustering(save_db=True)
    print_cluster_summary_report(results)

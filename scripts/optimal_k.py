import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

def find_optimal_k(csv_path="../data/processed/college_profiles_features.csv", max_k=6):
    df = pd.read_csv(csv_path)
    feature_cols = ["consistency_score", "tech_breadth_score",
                     "project_activity_score", "collaboration_score"]
    X = df[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)  # silhouette needs at least 2 clusters

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"k={k}: inertia={km.inertia_:.2f}, silhouette={score:.3f}")

    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"\n✅ Best k by silhouette score: {best_k}")

    # Plot both metrics side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, marker='o')
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouette_scores, marker='o', color='green')
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")

    os.makedirs("../outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("../outputs/optimal_k_analysis.png", dpi=150)
    print("Saved plot to outputs/optimal_k_analysis.png")

    return best_k

if __name__ == "__main__":
    find_optimal_k()
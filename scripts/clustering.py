import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_clustering(csv_path="college_profiles_features.csv", n_clusters=3):
    df = pd.read_csv(csv_path)
    
    # Features to cluster on
    feature_cols = ["consistency_score", "tech_breadth_score", 
                     "project_activity_score", "collaboration_score"]
    X = df[feature_cols]
    
    # Standardize features (important for K-Means - puts all scores on equal footing)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    
    # Label clusters by their average overall_skill_score (so labels make sense)
    cluster_avg = df.groupby("cluster")["overall_skill_score"].mean().sort_values()
    
    tier_names = ["Beginner", "Intermediate", "Advanced"]
    if n_clusters != 3:
        tier_names = [f"Tier {i}" for i in range(n_clusters)]
    
    cluster_to_tier = {cluster_id: tier_names[i] for i, cluster_id in enumerate(cluster_avg.index)}
    df["skill_tier"] = df["cluster"].map(cluster_to_tier)
    
    return df

if __name__ == "__main__":
    df = run_clustering()
    
    display_cols = ["username", "overall_skill_score", "skill_tier"]
    print(df[display_cols].sort_values("overall_skill_score", ascending=False))
    
    df.to_csv("college_profiles_clustered.csv", index=False)
    print(f"\n✅ Saved to college_profiles_clustered.csv")
    
    # Quick summary for your presentation
    print("\n--- Cluster Summary ---")
    print(df.groupby("skill_tier")[["consistency_score", "tech_breadth_score", 
                                      "project_activity_score", "collaboration_score"]].mean().round(2))
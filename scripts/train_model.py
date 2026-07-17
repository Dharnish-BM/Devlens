import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_save_model(csv_path="college_profiles_features.csv"):
    df = pd.read_csv(csv_path)
    
    feature_cols = ["consistency_score", "tech_breadth_score", 
                     "project_activity_score", "collaboration_score"]
    X = df[feature_cols]
    
    # Fit scaler and save it - new usernames must be scaled the SAME way as training data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-Means and save it
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Figure out which cluster number = which tier name (same logic as before)
    df["cluster"] = kmeans.predict(X_scaled)
    cluster_avg = df.groupby("cluster")["overall_skill_score"].mean().sort_values()
    tier_names = ["Beginner", "Intermediate", "Advanced"]
    cluster_to_tier = {cluster_id: tier_names[i] for i, cluster_id in enumerate(cluster_avg.index)}
    
    # Save everything needed to score a NEW username later
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(cluster_to_tier, "cluster_labels.pkl")
    
    # We also need min/max of each raw feature (for normalizing NEW usernames the same way)
    raw_df = pd.read_csv("college_profiles_raw.csv")
    normalization_ranges = {
        "num_unique_languages": (raw_df["num_unique_languages"].min(), raw_df["num_unique_languages"].max()),
        "activity_raw": None,  # computed below
        "collab_raw": None
    }
    
    activity_raw = raw_df["non_fork_repos"] * 3 + raw_df["total_stars"] + raw_df["total_forks"]
    collab_raw = raw_df["pr_contributions_past_year"] * 2 + raw_df["issue_contributions_past_year"]
    normalization_ranges["activity_raw"] = (activity_raw.min(), activity_raw.max())
    normalization_ranges["collab_raw"] = (collab_raw.min(), collab_raw.max())
    
    joblib.dump(normalization_ranges, "normalization_ranges.pkl")
    
    print("✅ Model, scaler, and normalization ranges saved:")
    print("   - scaler.pkl")
    print("   - kmeans_model.pkl")
    print("   - cluster_labels.pkl")
    print("   - normalization_ranges.pkl")
    print(f"\nCluster → Tier mapping: {cluster_to_tier}")

if __name__ == "__main__":
    train_and_save_model()
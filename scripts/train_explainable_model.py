import pandas as pd
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

def train_full_pipeline(features_csv="../data/processed/college_profiles_features.csv",
                          raw_csv="../data/raw/college_profiles_raw.csv",
                          n_clusters=3):
    
    df = pd.read_csv(features_csv)
    feature_cols = ["consistency_score", "tech_breadth_score",
                     "project_activity_score", "collaboration_score"]
    X = df[feature_cols]

    # --- Step 1: K-Means clustering (unsupervised skill tiers) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_avg = df.groupby("cluster")["overall_skill_score"].mean().sort_values()
    tier_names = ["Beginner", "Intermediate", "Advanced"][:n_clusters]
    cluster_to_tier = {cid: tier_names[i] for i, cid in enumerate(cluster_avg.index)}
    df["skill_tier"] = df["cluster"].map(cluster_to_tier)

    # --- Step 2: XGBoost learns to PREDICT the tier from raw features ---
    # This replaces GDERS's rule-based set-intersection recommendation logic
    # with a trained, generalizable model.
    y = df["cluster"]  # predict cluster ID (numeric)
    
    xgb_model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X, y)

    train_accuracy = xgb_model.score(X, y)
    print(f"XGBoost training accuracy (fit to KMeans tiers): {train_accuracy:.2%}")

    # --- Step 3: Build a SHAP explainer for this model ---
    explainer = shap.TreeExplainer(xgb_model)

    # --- Save everything ---
    os.makedirs("../models", exist_ok=True)
    joblib.dump(scaler, "../models/scaler.pkl")
    joblib.dump(kmeans, "../models/kmeans_model.pkl")
    joblib.dump(cluster_to_tier, "../models/cluster_labels.pkl")
    joblib.dump(xgb_model, "../models/xgb_model.pkl")
    joblib.dump(explainer, "../models/shap_explainer.pkl")
    joblib.dump(feature_cols, "../models/feature_names.pkl")

    # Normalization ranges (same as before, needed to score new usernames)
    raw_df = pd.read_csv(raw_csv)
    activity_raw = raw_df["non_fork_repos"] * 3 + raw_df["total_stars"] + raw_df["total_forks"]
    collab_raw = raw_df["pr_contributions_past_year"] * 2 + raw_df["issue_contributions_past_year"]
    normalization_ranges = {
        "num_unique_languages": (raw_df["num_unique_languages"].min(), raw_df["num_unique_languages"].max()),
        "activity_raw": (activity_raw.min(), activity_raw.max()),
        "collab_raw": (collab_raw.min(), collab_raw.max())
    }
    joblib.dump(normalization_ranges, "../models/normalization_ranges.pkl")

    print("\n✅ Saved: scaler, kmeans_model, cluster_labels, xgb_model, shap_explainer, feature_names, normalization_ranges")
    print(f"\nCluster → Tier mapping: {cluster_to_tier}")

if __name__ == "__main__":
    train_full_pipeline()
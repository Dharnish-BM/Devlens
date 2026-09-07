import sys
sys.path.append('d:/GIT/Devlens')
import joblib
import pandas as pd
import numpy as np
from devlens.db.session import get_session
from devlens.db.models import ClusterAssignment
from devlens.features.feature_engineering import prepare_scaled_feature_matrix
from devlens.db.repository import get_all_current_features
from sklearn.metrics import silhouette_score

# 1. Inspect saved K-Means model artifact
km = joblib.load("models/artifacts/kmeans_model.joblib")
scaler = joblib.load("models/artifacts/scaler.joblib")

print("=== 1. PERSISTED K-MEANS MODEL ARTIFACT INSPECTION ===")
print("Model type:          ", type(km))
print("n_clusters:          ", km.n_clusters)
print("n_features_in_:      ", km.n_features_in_)
print("feature_names_in_:   ", getattr(km, "feature_names_in_", None))
print("cluster_centers shape:", km.cluster_centers_.shape)
print("scaler feature count:", scaler.n_features_in_)

# 2. Inspect Cluster assignments in DB
with get_session() as session:
    rows = session.query(ClusterAssignment.cluster_id).all()
    df_clust = pd.DataFrame(rows, columns=["cluster_id"])

print("\n=== 2. CURRENT CLUSTER SIZES IN DATABASE ===")
print(df_clust["cluster_id"].value_counts())

# 3. Compute silhouette score against current 41-feature matrix
with get_session() as session:
    unscaled_df = get_all_current_features(session)

scaled_df, _, _ = prepare_scaled_feature_matrix(unscaled_df, drop_zero_variance=True)
preds = km.predict(scaled_df)
sil = silhouette_score(scaled_df, preds)
print(f"\nSilhouette score with current {scaled_df.shape[1]}-feature matrix: {sil:.4f}")
print("Predicted cluster sizes:", pd.Series(preds).value_counts().to_dict())

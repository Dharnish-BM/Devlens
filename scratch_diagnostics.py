import sys
sys.path.append('d:/GIT/Devlens')

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from devlens.db.session import get_session
from devlens.db.repository import get_all_current_features
from devlens.features.feature_engineering import prepare_scaled_feature_matrix

with get_session() as session:
    df_raw = get_all_current_features(session)

scaled_df, scaler, dropped_cols = prepare_scaled_feature_matrix(df_raw, drop_zero_variance=True)

lang_cols = [
    'lang_diversity_entropy',
    'lang_unique_count',
    'lang_primary_is_c',
    'lang_primary_is_java_kotlin',
    'lang_primary_is_javascript',
    'lang_primary_is_python',
    'lang_frontend_signal',
    'lang_devops_signal',
    'lang_ml_signal'
]

# Ensure all 9 columns are present in scaled_df
available_lang_cols = [c for c in lang_cols if c in scaled_df.columns]
scaled_lang = scaled_df[available_lang_cols]

print(f"Language Feature Matrix Shape: {scaled_lang.shape}")
print("Features used:", available_lang_cols)
print("\n" + "="*80)
print(f"{'k':<5} {'Silhouette Score':<20} {'Min Cluster Size (N)':<25} {'Status':<15} {'Cluster Sizes (N)'}")
print("-" * 80)

for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_lang)
    sil = silhouette_score(scaled_lang, labels)
    
    counts = pd.Series(labels).value_counts().sort_index().tolist()
    min_size = min(counts)
    min_pct = min_size / len(scaled_lang) * 100
    valid_str = "VALID (>=5%)" if min_pct >= 5.0 else "INVALID (<5%)"
    
    counts_str = ", ".join([f"C{i}:{n}" for i, n in enumerate(counts)])
    print(f"{k:<5} {sil:<20.4f} {min_size} ({min_pct:.1f}%) {' ':<12} {valid_str:<15} {counts_str}")

print("="*80)

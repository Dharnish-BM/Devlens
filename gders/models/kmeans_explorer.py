"""
GDERS Exploratory K-Means Clustering Analysis Module.

Implements unsupervised TF-IDF K-Means clustering analysis across multiple K values,
evaluating inertia, silhouette scores, Adjusted Rand Index (ARI), and cluster purity
against reference gold annotations (exploratory diagnostic only).
"""

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

logger = logging.getLogger(__name__)


class GDERSKMeansExplorer:
    """Exploratory K-Means Clustering for GDERS comment corpus."""

    def __init__(
        self,
        max_features: int = 1500,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        random_seed: int = 42,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.random_seed = random_seed
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        )

    def evaluate_clustering(
        self,
        texts: List[str],
        primary_labels: List[str],
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate K-Means across k_values and compute cluster-to-reference agreement metrics."""
        if k_values is None:
            k_values = [5, 8, 10, 12]

        X = self.vectorizer.fit_transform(texts)
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        results = {}
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
            cluster_ids = km.fit_predict(X)

            sil_score = float(silhouette_score(X, cluster_ids)) if len(texts) > k else 0.0
            ari = float(adjusted_rand_score(primary_labels, cluster_ids))
            nmi = float(normalized_mutual_info_score(primary_labels, cluster_ids))

            # Cluster sizes and top terms
            cluster_sizes = collections.Counter(cluster_ids)
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]
            cluster_terms = {}
            for i in range(k):
                top_terms = [feature_names[ind] for ind in order_centroids[i, :8]]
                cluster_terms[f"cluster_{i}"] = {
                    "size": int(cluster_sizes.get(i, 0)),
                    "top_terms": top_terms,
                }

            results[f"k_{k}"] = {
                "k": k,
                "inertia": round(float(km.inertia_), 2),
                "silhouette_score": round(sil_score, 4),
                "adjusted_rand_index_vs_reference": round(ari, 4),
                "normalized_mutual_info_vs_reference": round(nmi, 4),
                "clusters": cluster_terms,
            }

        return {
            "exploratory_note": "K-Means cluster IDs are arbitrary unsupervised partitions and do not constitute ground truth expertise.",
            "k_evaluations": results,
        }

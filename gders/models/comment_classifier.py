"""
GDERS Multi-Label Comment Classifier Module.

Implements classical multi-label classifiers (TF-IDF + LogisticRegression / LinearSVC)
with One-vs-Rest strategy, stratified cross-validation evaluation, and top-term inspection.
"""

import collections
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from gders.config import DEFAULT_CONFIG, GDERSConfig

logger = logging.getLogger(__name__)


class CommentClassifier:
    """Multi-Label Review Comment Classifier using TF-IDF and One-vs-Rest estimators."""

    def __init__(
        self,
        config: Optional[GDERSConfig] = None,
        model_type: str = "logistic_regression",  # 'logistic_regression' or 'linear_svm'
        max_features: int = 1500,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 1.0,
        class_weight: str = "balanced",
        random_seed: int = 42,
    ):
        self.config = config or DEFAULT_CONFIG
        self.model_type = model_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.C = C
        self.class_weight = class_weight
        self.random_seed = random_seed

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        )
        self.mlb = MultiLabelBinarizer()
        self.model: Optional[OneVsRestClassifier] = None
        self.categories: List[str] = []
        self.is_trained: bool = False

    def _build_estimator(self):
        """Construct the base linear estimator."""
        if self.model_type == "logistic_regression":
            base = LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=1000,
                random_state=self.random_seed,
                solver="liblinear",
            )
        elif self.model_type == "linear_svm":
            base = LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=2000,
                random_state=self.random_seed,
                dual=False,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return OneVsRestClassifier(base)

    def fit(self, texts: List[str], label_lists: List[List[str]]) -> "CommentClassifier":
        """Fit TF-IDF vectorizer and One-vs-Rest classifier on multi-label training data."""
        if not texts or not label_lists:
            raise ValueError("Texts and label_lists cannot be empty.")

        # Multi-label binarization
        Y = self.mlb.fit_transform(label_lists)
        self.categories = list(self.mlb.classes_)

        # Vectorization
        X = self.vectorizer.fit_transform(texts)

        # Train model
        self.model = self._build_estimator()
        self.model.fit(X, Y)
        self.is_trained = True
        return self

    def predict(self, texts: List[str]) -> List[List[str]]:
        """Predict list of category labels for input texts."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Classifier is not trained yet.")
        X = self.vectorizer.transform(texts)
        Y_pred = self.model.predict(X)
        return self.mlb.inverse_transform(Y_pred)

    def predict_binary_matrix(self, texts: List[str]) -> np.ndarray:
        """Predict binary indicator matrix [n_samples, n_classes]."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Classifier is not trained yet.")
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_decision_scores(self, texts: List[str]) -> np.ndarray:
        """Predict decision function / probability matrix."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Classifier is not trained yet.")
        X = self.vectorizer.transform(texts)
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        elif hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def get_top_features_per_category(self, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Inspect top positive TF-IDF n-grams for each category."""
        if not self.is_trained or self.model is None:
            return {}

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_features: Dict[str, List[Tuple[str, float]]] = {}

        for idx, cat in enumerate(self.categories):
            estimator = self.model.estimators_[idx]
            coefs = getattr(estimator, "coef_", None)
            if coefs is not None and len(coefs) > 0:
                coef_arr = coefs[0] if coefs.ndim > 1 else coefs
                top_indices = np.argsort(coef_arr)[::-1][:top_n]
                top_terms = [(feature_names[i], round(float(coef_arr[i]), 4)) for i in top_indices]
                top_features[cat] = top_terms
            else:
                top_features[cat] = []

        return top_features

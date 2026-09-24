"""
Unit tests for GDERS Phase 5B Model Comparison, Cross-Validation, and Error Analysis.
"""

import numpy as np
from gders.config import GDERSConfig
from gders.evaluation.model_evaluator import (
    compute_multilabel_metrics,
    run_cross_validation_experiment,
)
from gders.models.comment_classifier import CommentClassifier
from gders.models.kmeans_explorer import GDERSKMeansExplorer


def test_multilabel_metrics_computation():
    y_true = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
    ])
    y_pred = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ])
    classes = ["CAT_A", "CAT_B", "CAT_C"]

    metrics = compute_multilabel_metrics(y_true, y_pred, classes)

    assert "micro_f1" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_category" in metrics
    assert metrics["micro_f1"] > 0.7
    assert abs(metrics["subset_accuracy"] - 2 / 3) < 1e-3


def test_comment_classifier_fit_and_predict():
    texts = [
        "Please add unit test and mock assertion for this method",
        "Fix null pointer dereference bug in loop condition",
        "Optimize memory allocation using hashset lookup",
        "Rename variable to follow style naming convention",
    ]
    labels = [
        ["TESTING_QUALITY"],
        ["BUG_LOGIC"],
        ["PERF_OPTIMIZATION"],
        ["CODE_STYLE"],
    ]

    clf = CommentClassifier(model_type="logistic_regression", C=1.0, min_df=1)
    clf.fit(texts, labels)

    assert clf.is_trained is True
    assert len(clf.categories) == 4

    preds = clf.predict(["Need to add regression test with assertions."])
    assert len(preds) == 1
    assert isinstance(preds[0], tuple)

    top_feats = clf.get_top_features_per_category(top_n=3)
    assert "TESTING_QUALITY" in top_feats
    assert "BUG_LOGIC" in top_feats


def test_cross_validation_experiment_execution():
    texts = [
        "Please add a unit test for this handler",
        "Fix null check crash in loop",
        "Optimize memory allocation in database query",
        "Update docstring explaining the return format",
        "Nit: rename this variable to camelCase",
        "Add integration test verifying API status 200",
        "Unhandled exception causes panic on shutdown",
        "Use indexed lookup to avoid O(N^2) latency",
        "Document exception thrown on network timeout",
        "Remove unused imports and clean formatting",
    ]
    labels = [
        ["TESTING_QUALITY"],
        ["BUG_LOGIC"],
        ["PERF_OPTIMIZATION", "DATA_MANAGEMENT"],
        ["DOCUMENTATION"],
        ["CODE_STYLE"],
        ["TESTING_QUALITY"],
        ["BUG_LOGIC"],
        ["PERF_OPTIMIZATION"],
        ["DOCUMENTATION"],
        ["CODE_STYLE"],
    ]
    records = [{"comment_id": i + 1, "repository": "test/repo", "original_body": texts[i]} for i in range(10)]

    summary, errors = run_cross_validation_experiment(
        texts=texts,
        labels=labels,
        records=records,
        model_type="linear_svm",
        n_splits=3,
        random_seed=42,
    )

    assert "micro_f1_mean" in summary
    assert "macro_f1_mean" in summary
    assert "per_category" in summary
    assert len(summary["folds"]) == 3
    assert isinstance(errors, list)


def test_kmeans_exploratory_clustering():
    texts = [
        "unit test assertion mock fixture",
        "regression test verification spec",
        "null pointer crash bug fix",
        "exception unhandled nil error",
        "optimize memory allocation benchmark",
        "docstring documentation javadoc comment",
    ]
    primary_labels = ["TESTING", "TESTING", "BUG", "BUG", "PERF", "DOCS"]

    explorer = GDERSKMeansExplorer(random_seed=42)
    results = explorer.evaluate_clustering(texts, primary_labels, k_values=[2, 3])

    assert "k_evaluations" in results
    assert "k_2" in results["k_evaluations"]
    assert "k_3" in results["k_evaluations"]
    assert "inertia" in results["k_evaluations"]["k_2"]
    assert "silhouette_score" in results["k_evaluations"]["k_2"]

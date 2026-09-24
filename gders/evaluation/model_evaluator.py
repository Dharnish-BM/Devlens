"""
GDERS Model Evaluation, Cross-Validation, & Error Analysis Module.

Implements multi-label evaluation metrics (micro/macro/weighted F1, precision, recall,
Hamming loss, subset accuracy), multi-label cross-validation, and error record extraction.
"""

import collections
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import KFold

from gders.models.comment_classifier import CommentClassifier

logger = logging.getLogger(__name__)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
) -> Dict[str, Any]:
    """Compute comprehensive multi-label evaluation metrics."""
    # Global multi-label metrics
    micro_p = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    micro_r = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    h_loss = float(hamming_loss(y_true, y_pred))
    exact_match = float(accuracy_score(y_true, y_pred))

    # Per-class metrics
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_support = np.sum(y_true, axis=0)

    class_metrics = {}
    for idx, c_name in enumerate(classes):
        class_metrics[c_name] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1": round(float(per_class_f1[idx]), 4),
            "support": int(per_class_support[idx]),
        }

    return {
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "hamming_loss": round(h_loss, 4),
        "subset_accuracy": round(exact_match, 4),
        "per_category": class_metrics,
    }


def run_cross_validation_experiment(
    texts: List[str],
    labels: List[List[str]],
    records: List[Dict[str, Any]],
    model_type: str = "logistic_regression",
    n_splits: int = 5,
    random_seed: int = 42,
    C: float = 1.0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run deterministic 5-fold cross-validation and extract error records."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_metrics: List[Dict[str, Any]] = []

    # Get universe of classes
    all_classes_set = sorted(list({lbl for sub in labels for lbl in sub}))
    n_samples = len(texts)

    # Storage for out-of-fold predictions
    oof_y_true = []
    oof_y_pred = []
    oof_classes = []
    error_analysis_records: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(texts), start=1):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        val_records = [records[i] for i in val_idx]

        clf = CommentClassifier(
            model_type=model_type,
            C=C,
            random_seed=random_seed + fold_idx,
        )
        clf.fit(train_texts, train_labels)

        # Align validation binary matrices to common class list
        y_val_true = clf.mlb.transform(val_labels)
        y_val_pred = clf.predict_binary_matrix(val_texts)
        scores = clf.predict_decision_scores(val_texts)

        metrics = compute_multilabel_metrics(y_val_true, y_val_pred, clf.categories)
        fold_metrics.append(metrics)

        # Extract misclassifications for error analysis
        for sample_i in range(len(val_texts)):
            true_l = val_labels[sample_i]
            pred_l = list(clf.mlb.inverse_transform(y_val_pred[sample_i : sample_i + 1])[0])

            is_error = set(true_l) != set(pred_l)
            if is_error:
                rec_meta = val_records[sample_i]
                err_type = (
                    "false_positive_only" if set(true_l).issubset(set(pred_l))
                    else ("false_negative_only" if set(pred_l).issubset(set(true_l))
                    else "mixed_confusion")
                )
                error_analysis_records.append({
                    "comment_id": rec_meta.get("comment_id"),
                    "repository": rec_meta.get("repository"),
                    "fold": fold_idx,
                    "original_body": rec_meta.get("original_body"),
                    "processed_text": val_texts[sample_i],
                    "true_labels": true_l,
                    "predicted_labels": pred_l,
                    "error_type": err_type,
                })

    # Compute mean and standard deviation across folds
    metric_keys = [
        "micro_precision", "micro_recall", "micro_f1",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_f1", "hamming_loss", "subset_accuracy"
    ]
    summary = {}
    for mk in metric_keys:
        vals = [fm[mk] for fm in fold_metrics]
        summary[f"{mk}_mean"] = round(float(np.mean(vals)), 4)
        summary[f"{mk}_std"] = round(float(np.std(vals)), 4)

    # Average per-category metrics across folds
    per_cat_summary = {}
    for cat in all_classes_set:
        cat_f1s = [fm["per_category"].get(cat, {}).get("f1", 0.0) for fm in fold_metrics]
        cat_precs = [fm["per_category"].get(cat, {}).get("precision", 0.0) for fm in fold_metrics]
        cat_recs = [fm["per_category"].get(cat, {}).get("recall", 0.0) for fm in fold_metrics]
        cat_supports = [fm["per_category"].get(cat, {}).get("support", 0) for fm in fold_metrics]

        per_cat_summary[cat] = {
            "f1_mean": round(float(np.mean(cat_f1s)), 4),
            "f1_std": round(float(np.std(cat_f1s)), 4),
            "precision_mean": round(float(np.mean(cat_precs)), 4),
            "recall_mean": round(float(np.mean(cat_recs)), 4),
            "total_support": int(np.sum(cat_supports)),
        }

    summary["per_category"] = per_cat_summary
    summary["folds"] = fold_metrics
    summary["total_error_count"] = len(error_analysis_records)

    return summary, error_analysis_records

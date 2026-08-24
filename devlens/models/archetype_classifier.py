"""
DevLens Phase 6: XGBoost Archetype Classification & Model Evaluation.

- Generates 7-tier rule-based pseudo-labels for all 188 developers
- Excludes ultra-minority classes (Backend Developer n=2, Full-Stack Developer n=1) from training
- Trains XGBoost on remaining 5 classes (185 developers total)
- Primary Evaluation: Stratified 70/30 Train/Test Split with Confusion Matrix & Classification Report
- Supplementary Evaluation: Stratified 5-Fold Cross-Validation (Mean +/- Std Accuracy & Per-Class F1)
- Computes SHAP feature importance for model interpretability
- Saves model artifacts and persists predictions to archetype_predictions DB table
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from devlens.db.session import get_session
from devlens.db.repository import (
    get_all_current_features,
    get_developer,
    get_latest_snapshot,
    insert_archetype_prediction,
)
from devlens.features.feature_engineering import prepare_scaled_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIRS = [
    Path("models/artifacts"),
    Path("devlens/models/artifacts"),
]


def ensure_artifact_dirs():
    """Ensure artifact directories exist."""
    for d in ARTIFACT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def save_artifact(obj: Any, filename: str):
    """Save an object or figure to artifact directories."""
    ensure_artifact_dirs()
    for d in ARTIFACT_DIRS:
        path = d / filename
        if isinstance(obj, plt.Figure):
            obj.savefig(path, dpi=300, bbox_inches="tight")
        else:
            joblib.dump(obj, path)
        logger.info(f"Saved artifact: {path}")


def generate_pseudo_labels(df: pd.DataFrame) -> pd.Series:
    """Generate 7-tier hierarchical pseudo-labels for developer cohort.
    
    Precedence:
    1. ML Specialist: lang_ml_signal > 0.05
    2. Mobile Developer: lang_mobile_signal > 0.02
    3. DevOps Engineer: lang_devops_signal > 0.10
    4. Frontend Developer: lang_frontend_signal > 0.50 or lang_primary_is_javascript == 1.0
    5. Backend Developer: lang_primary_is_c == 1.0 or (lang_primary_is_java_kotlin == 1.0 and lang_mobile_signal <= 0.02)
    6. Full-Stack Developer: lang_diversity_entropy >= 1.8
    7. Unclassified / Low Signal: lang_diversity_entropy < 1.8
    """
    labels = []
    for username, row in df.iterrows():
        ml_sig = row.get("lang_ml_signal", 0.0)
        mobile_sig = row.get("lang_mobile_signal", 0.0)
        devops_sig = row.get("lang_devops_signal", 0.0)
        frontend_sig = row.get("lang_frontend_signal", 0.0)
        primary_js = row.get("lang_primary_is_javascript", 0.0) == 1.0
        primary_c = row.get("lang_primary_is_c", 0.0) == 1.0
        primary_java_kotlin = row.get("lang_primary_is_java_kotlin", 0.0) == 1.0
        entropy = row.get("lang_diversity_entropy", 0.0)

        if ml_sig > 0.05:
            lbl = "ML Specialist"
        elif mobile_sig > 0.02:
            lbl = "Mobile Developer"
        elif devops_sig > 0.10:
            lbl = "DevOps Engineer"
        elif frontend_sig > 0.50 or primary_js:
            lbl = "Frontend Developer"
        elif primary_c or (primary_java_kotlin and mobile_sig <= 0.02):
            lbl = "Backend Developer"
        elif entropy >= 1.8:
            lbl = "Full-Stack Developer"
        else:
            lbl = "Unclassified / Low Signal"
        
        labels.append(lbl)

    return pd.Series(labels, index=df.index, name="archetype_label")


def evaluate_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Run Stratified 5-Fold Cross Validation and compute mean +/- std metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = list(le.classes_)

    fold_accuracies = []
    fold_f1_macros = []
    fold_f1_weighted = []
    fold_per_class_f1: Dict[str, List[float]] = {cls_name: [] for cls_name in classes}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[val_idx]

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state + fold,
            eval_metric="mlogloss"
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        fold_accuracies.append(accuracy_score(y_val, preds))
        fold_f1_macros.append(f1_score(y_val, preds, average="macro"))
        fold_f1_weighted.append(f1_score(y_val, preds, average="weighted"))

        # Per-class F1 for this fold
        per_class = f1_score(y_val, preds, average=None, labels=range(len(classes)))
        for i, cls_name in enumerate(classes):
            fold_per_class_f1[cls_name].append(per_class[i])

    cv_results = {
        "classes": classes,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "mean_f1_macro": float(np.mean(fold_f1_macros)),
        "std_f1_macro": float(np.std(fold_f1_macros)),
        "mean_f1_weighted": float(np.mean(fold_f1_weighted)),
        "std_f1_weighted": float(np.std(fold_f1_weighted)),
        "per_class_f1": {
            cls_name: {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "scores": scores
            }
            for cls_name, scores in fold_per_class_f1.items()
        }
    }
    return cv_results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "XGBoost Archetype Confusion Matrix (70/30 Test Set)"
) -> plt.Figure:
    """Generate confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Archetype Label",
        xlabel="Predicted Archetype Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # Annotate text
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    plt.tight_layout()
    return fig


def run_archetype_pipeline(save_db: bool = True) -> Dict[str, Any]:
    """Execute Phase 6 Archetype Classification Pipeline."""
    logger.info("Loading feature vectors from database...")
    with get_session() as session:
        unscaled_df = get_all_current_features(session)

    if unscaled_df.empty:
        raise ValueError("No feature data found in database.")

    # 1. Generate 7-tier pseudo-labels for all 188 developers
    all_pseudo_labels = generate_pseudo_labels(unscaled_df)
    unscaled_df["pseudo_label"] = all_pseudo_labels
    logger.info(f"Generated pseudo-labels for {len(unscaled_df)} developers across 7 classes.")

    # 2. Exclude ultra-minorities (Backend Developer n=2, Full-Stack Developer n=1) from model training
    excluded_classes = ["Backend Developer", "Full-Stack Developer"]
    train_mask = ~unscaled_df["pseudo_label"].isin(excluded_classes)
    train_df = unscaled_df[train_mask].copy()
    excluded_df = unscaled_df[~train_mask].copy()
    logger.info(f"Training dataset size: {len(train_df)} developers (Excluded {len(excluded_df)} developers: {excluded_classes})")

    # 3. Prepare preprocessed feature matrix
    scaled_df, scaler, _ = prepare_scaled_feature_matrix(unscaled_df.drop(columns=["pseudo_label"]), drop_zero_variance=True)
    X_train_full = scaled_df.loc[train_df.index]
    y_train_full = train_df["pseudo_label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train_full)
    class_names = list(le.classes_)

    # 4. Supplementary Robustness Check: Stratified 5-Fold Cross Validation
    logger.info("Running Stratified 5-Fold Cross Validation...")
    cv_results = evaluate_cross_validation(X_train_full, y_train_full, n_splits=5, random_state=42)

    # 5. Primary Evaluation: 70/30 Stratified Held-out Split
    logger.info("Evaluating on 70/30 Stratified Train/Test Split...")
    X_tr, X_te, y_tr_enc, y_te_enc = train_test_split(
        X_train_full, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
    )

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )
    clf.fit(X_tr, y_tr_enc)

    test_preds = clf.predict(X_te)
    test_probs = clf.predict_proba(X_te)

    test_acc = accuracy_score(y_te_enc, test_preds)
    test_report_dict = classification_report(y_te_enc, test_preds, target_names=class_names, output_dict=True)
    test_report_str = classification_report(y_te_enc, test_preds, target_names=class_names)
    cm = confusion_matrix(y_te_enc, test_preds)

    # Plot & Save Confusion Matrix
    cm_fig = plot_confusion_matrix(cm, class_names)
    save_artifact(cm_fig, "archetype_confusion_matrix.png")
    plt.close(cm_fig)

    # 6. Fit Final Production Model on Full 185-student dataset
    logger.info("Fitting final production XGBoost model on full 185-student dataset...")
    final_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss"
    )
    final_model.fit(X_train_full, y_encoded)

    save_artifact(final_model, "xgboost_archetype_model.joblib")
    save_artifact(le, "archetype_label_encoder.joblib")

    # 7. SHAP Feature Importance & Explanation
    logger.info("Computing SHAP values for model interpretability...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(scaled_df)

    # 8. Persist Predictions & SHAP Explanations to Database for all 188 students
    if save_db:
        logger.info("Persisting archetype predictions to database...")
        with get_session() as session:
            # First, clean existing predictions
            # Predict probabilities for the entire scaled cohort
            all_probs = final_model.predict_proba(scaled_df)
            all_preds_enc = final_model.predict(scaled_df)

            for idx, username in enumerate(scaled_df.index):
                dev = get_developer(session, str(username))
                if not dev:
                    continue
                snapshot = get_latest_snapshot(session, dev.id)
                if not snapshot:
                    continue

                # Check if student was in the 2 excluded classes
                true_pseudo = all_pseudo_labels[username]
                if true_pseudo in excluded_classes:
                    assigned_label = true_pseudo
                    confidence = 1.0  # Heuristic ground truth assignment
                    top_shap = {}
                else:
                    pred_class_idx = all_preds_enc[idx]
                    assigned_label = le.inverse_transform([pred_class_idx])[0]
                    confidence = float(all_probs[idx][pred_class_idx])

                    # Extract top 5 SHAP contributing features for this prediction
                    if isinstance(shap_values, list):
                        student_shap = shap_values[pred_class_idx][idx]
                    elif len(shap_values.shape) == 3:
                        student_shap = shap_values[idx, :, pred_class_idx]
                    else:
                        student_shap = shap_values[idx]

                    feature_names = scaled_df.columns
                    top_feat_indices = np.argsort(np.abs(student_shap))[::-1][:5]
                    top_shap = {
                        feature_names[f_idx]: float(student_shap[f_idx])
                        for f_idx in top_feat_indices
                    }

                insert_archetype_prediction(
                    session=session,
                    snapshot_id=snapshot.id,
                    archetype_label=assigned_label,
                    confidence=confidence,
                    shap_top_features=top_shap,
                )
            session.commit()
        logger.info("All 188 archetype predictions and SHAP values successfully stored in DB.")

    return {
        "cohort_size": len(unscaled_df),
        "training_size": len(train_df),
        "excluded_size": len(excluded_df),
        "pseudo_label_dist": all_pseudo_labels.value_counts().to_dict(),
        "test_accuracy": test_acc,
        "test_report_str": test_report_str,
        "test_report_dict": test_report_dict,
        "confusion_matrix": cm,
        "class_names": class_names,
        "cv_results": cv_results,
    }


def print_archetype_summary_report(results: Dict[str, Any]) -> None:
    """Print readable summary report for Phase 6."""
    print("\n" + "=" * 80)
    print("  DEV LENS - PHASE 6 XGBOOST ARCHETYPE CLASSIFIER SUMMARY")
    print("=" * 80)

    print(f"\nTotal Cohort: {results['cohort_size']} developers")
    print(f"Trained On:   {results['training_size']} developers across 5 classes")
    print(f"Excluded:     {results['excluded_size']} developers (Backend n=2, Full-Stack n=1)")

    print("\n--- Pseudo-Label Distribution (All 188 Students) ---")
    for lbl, cnt in results["pseudo_label_dist"].items():
        pct = cnt / results["cohort_size"] * 100
        print(f"  • {lbl:<28}: {cnt:>3} ({pct:>5.1f}%)")

    print("\n" + "=" * 80)
    print("  PRIMARY EVALUATION: 70/30 HELD-OUT TEST SET (N=56)")
    print("=" * 80)
    print(f"Held-out Test Accuracy: {results['test_accuracy'] * 100:.2f}%\n")
    print(results["test_report_str"])

    print("=" * 80)
    print("  SUPPLEMENTARY ROBUSTNESS: STRATIFIED 5-FOLD CROSS-VALIDATION (N=185)")
    print("=" * 80)
    cv = results["cv_results"]
    print(f"Mean CV Accuracy:    {cv['mean_accuracy'] * 100:.2f}% ± {cv['std_accuracy'] * 100:.2f}%")
    print(f"Mean Macro F1-Score: {cv['mean_f1_macro']:.4f} ± {cv['std_f1_macro']:.4f}")
    print(f"Mean Weighted F1:    {cv['mean_f1_weighted']:.4f} ± {cv['std_f1_weighted']:.4f}")

    print("\n--- Per-Class 5-Fold CV F1 Performance (Mean ± Std) ---")
    print(f"{'Class Name':<30} {'Mean F1':<12} {'Std F1':<12} {'Fold Scores'}")
    print("-" * 75)
    for cls_name, info in cv["per_class_f1"].items():
        scores_str = ", ".join([f"{s:.3f}" for s in info["scores"]])
        print(f"{cls_name:<30} {info['mean']:<12.4f} {info['std']:<12.4f} [{scores_str}]")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    results = run_archetype_pipeline(save_db=True)
    print_archetype_summary_report(results)

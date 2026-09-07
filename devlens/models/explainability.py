"""
DevLens Phase 6 Explainability Module: SHAP Interpretability & Label Consistency.

- Explains XGBoost archetype classifier predictions using shap.TreeExplainer
- Handles exact 3D ndarray (N, 41, 5) SHAP output structure with strict assertions
- Computes top 5 SHAP contributing features per developer
- Computes label_consistency_report(): % of SHAP magnitude from 8 labeling features vs 33 behavioral features
- Generates and saves SHAP summary plots to models/artifacts/shap_summary.png
- Persists all 188 developer explanations to the database
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from devlens.db.models import ArchetypePrediction, Developer, Snapshot
from devlens.db.repository import (
    get_all_current_features,
    get_developer,
    get_latest_snapshot,
    insert_archetype_prediction,
)
from devlens.db.session import get_session
from devlens.features.feature_engineering import prepare_scaled_feature_matrix
from devlens.models.archetype_classifier import generate_pseudo_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIRS = [
    Path("models/artifacts"),
    Path("devlens/models/artifacts"),
]

LABELING_SOURCE_FEATURES = [
    "lang_ml_signal",
    "lang_mobile_signal",
    "lang_devops_signal",
    "lang_frontend_signal",
    "lang_primary_is_javascript",
    "lang_primary_is_c",
    "lang_primary_is_java_kotlin",
    "lang_diversity_entropy",
]


def ensure_artifact_dirs():
    """Ensure artifact directories exist."""
    for d in ARTIFACT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def save_artifact(obj: Any, filename: str):
    """Save plot or object to artifact directories."""
    ensure_artifact_dirs()
    for d in ARTIFACT_DIRS:
        path = d / filename
        if isinstance(obj, plt.Figure):
            obj.savefig(path, dpi=300, bbox_inches="tight")
        else:
            joblib.dump(obj, path)
        logger.info(f"Saved artifact: {path}")


def load_model_and_artifacts() -> Tuple[Any, Any]:
    """Load the deployed Phase 6 XGBoost model and LabelEncoder."""
    model_path = Path("models/artifacts/xgboost_archetype_model.joblib")
    le_path = Path("models/artifacts/archetype_label_encoder.joblib")

    if not model_path.exists() or not le_path.exists():
        raise FileNotFoundError(
            f"Model or label encoder not found at {model_path}. Run devlens.models.archetype_classifier first."
        )

    model = joblib.load(model_path)
    le = joblib.load(le_path)
    return model, le


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Instantiate and return SHAP TreeExplainer for the trained XGBoost model."""
    return shap.TreeExplainer(model)


def extract_shap_values(
    explainer: shap.TreeExplainer,
    scaled_df: pd.DataFrame,
    expected_classes: int = 5
) -> np.ndarray:
    """Compute and validate SHAP values structure.
    
    Validates that the SHAP version returns a 3D ndarray of shape (N, num_features, num_classes).
    Fails loudly if shape or type does not match expectations.
    """
    shap_vals = explainer.shap_values(scaled_df)
    n_samples, n_features = scaled_df.shape

    # SHAP 0.51+ on multiclass XGBoost returns ndarray with shape (N, n_features, n_classes)
    if isinstance(shap_vals, list):
        # Fallback if list of arrays is returned by different backend
        assert len(shap_vals) == expected_classes, (
            f"SHAP list output length mismatch: expected {expected_classes}, got {len(shap_vals)}"
        )
        shap_3d = np.stack(shap_vals, axis=-1)
    elif isinstance(shap_vals, np.ndarray):
        shap_3d = shap_vals
    else:
        raise TypeError(f"Unexpected SHAP values type: {type(shap_vals)}")

    # Strict assertion check
    assert shap_3d.ndim == 3, f"Expected 3D SHAP array, got {shap_3d.ndim}D array with shape {shap_3d.shape}"
    assert shap_3d.shape == (n_samples, n_features, expected_classes), (
        f"SHAP shape mismatch: expected ({n_samples}, {n_features}, {expected_classes}), got {shap_3d.shape}"
    )

    logger.info(f"Verified SHAP values 3D array shape: {shap_3d.shape}")
    return shap_3d


def explain_prediction(
    feature_vector: pd.Series,
    model: Any,
    explainer: shap.TreeExplainer,
    le: Any,
    top_n: int = 5,
    target_class: Optional[str] = None
) -> Dict[str, Any]:
    """Explain a single developer feature vector.
    
    Returns:
        Dict with predicted_class, confidence, and top_n SHAP contributing features (name: shap_value).
    """
    df_single = pd.DataFrame([feature_vector])
    probs = model.predict_proba(df_single)[0]
    
    if target_class and target_class in le.classes_:
        pred_label = target_class
        pred_idx = int(np.where(le.classes_ == target_class)[0][0])
        confidence = float(probs[pred_idx])
    else:
        pred_idx = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = float(probs[pred_idx])

    shap_single = explainer.shap_values(df_single)
    if isinstance(shap_single, list):
        sample_class_shap = shap_single[pred_idx][0]
    elif isinstance(shap_single, np.ndarray) and shap_single.ndim == 3:
        sample_class_shap = shap_single[0, :, pred_idx]
    else:
        sample_class_shap = shap_single[0]

    feature_names = df_single.columns
    top_indices = np.argsort(np.abs(sample_class_shap))[::-1][:top_n]

    shap_top_features = {
        str(feature_names[i]): float(sample_class_shap[i])
        for i in top_indices
    }

    return {
        "predicted_class": pred_label,
        "confidence": confidence,
        "shap_top_features": shap_top_features,
    }


def label_consistency_report(
    shap_3d: np.ndarray,
    preds_encoded: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    excluded_counts: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Compute % of total SHAP magnitude coming from 8 labeling vs 33 behavioral features.
    
    Computes per-class and overall dataset totals.
    """
    labeling_indices = [i for i, f in enumerate(feature_names) if f in LABELING_SOURCE_FEATURES]
    behavioral_indices = [i for i, f in enumerate(feature_names) if f not in LABELING_SOURCE_FEATURES]

    n_samples = shap_3d.shape[0]
    class_stats: Dict[str, Dict[str, float]] = {
        cls: {"labeling_mag": 0.0, "behavioral_mag": 0.0, "total_mag": 0.0, "count": 0}
        for cls in class_names
    }

    total_labeling_mag = 0.0
    total_behavioral_mag = 0.0

    for i in range(n_samples):
        cls_idx = preds_encoded[i]
        cls_name = class_names[cls_idx]
        sample_shap = shap_3d[i, :, cls_idx]

        labeling_mag = float(np.sum(np.abs(sample_shap[labeling_indices])))
        behavioral_mag = float(np.sum(np.abs(sample_shap[behavioral_indices])))
        total_mag = labeling_mag + behavioral_mag

        class_stats[cls_name]["labeling_mag"] += labeling_mag
        class_stats[cls_name]["behavioral_mag"] += behavioral_mag
        class_stats[cls_name]["total_mag"] += total_mag
        class_stats[cls_name]["count"] += 1

        total_labeling_mag += labeling_mag
        total_behavioral_mag += behavioral_mag

    grand_total_mag = total_labeling_mag + total_behavioral_mag

    # Per-class summary percentages
    report_rows = []
    for cls in class_names:
        stats = class_stats[cls]
        c_tot = stats["total_mag"]
        lbl_pct = (stats["labeling_mag"] / c_tot * 100) if c_tot > 0 else 0.0
        beh_pct = (stats["behavioral_mag"] / c_tot * 100) if c_tot > 0 else 0.0
        report_rows.append({
            "class_name": cls,
            "count": stats["count"],
            "labeling_pct": lbl_pct,
            "behavioral_pct": beh_pct,
            "avg_shap_magnitude": c_tot / max(stats["count"], 1),
        })

    # Include heuristic rule classes (Backend Developer, Full-Stack Developer) as 100% rule-derived
    if excluded_counts:
        for ex_cls, ex_cnt in excluded_counts.items():
            if ex_cnt > 0:
                report_rows.append({
                    "class_name": f"{ex_cls} (Heuristic)",
                    "count": ex_cnt,
                    "labeling_pct": 100.0,
                    "behavioral_pct": 0.0,
                    "avg_shap_magnitude": 0.0,
                })

    overall_labeling_pct = (total_labeling_mag / grand_total_mag * 100) if grand_total_mag > 0 else 0.0
    overall_behavioral_pct = (total_behavioral_mag / grand_total_mag * 100) if grand_total_mag > 0 else 0.0

    return {
        "overall_labeling_pct": overall_labeling_pct,
        "overall_behavioral_pct": overall_behavioral_pct,
        "total_samples_analyzed": n_samples,
        "per_class_summary": report_rows,
    }


def generate_shap_summary_plot(
    shap_3d: np.ndarray,
    scaled_df: pd.DataFrame,
    class_names: List[str]
) -> plt.Figure:
    """Generate SHAP multi-class summary plot across all 188 developers."""
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(
        [shap_3d[:, :, i] for i in range(len(class_names))],
        scaled_df,
        class_names=class_names,
        show=False,
        max_display=15
    )
    plt.title("SHAP Feature Importance & Contribution across All 188 Developers", fontsize=13, pad=15)
    plt.tight_layout()
    return fig


def run_explainability_pipeline(save_db: bool = True) -> Dict[str, Any]:
    """Execute end-to-end SHAP explainability analysis."""
    logger.info("Loading feature vectors and Phase 6 model...")
    with get_session() as session:
        unscaled_df = get_all_current_features(session)

    if unscaled_df.empty:
        raise ValueError("No feature data found in database.")

    model, le = load_model_and_artifacts()
    class_names = list(le.classes_)

    # Scale feature matrix exactly matching training
    scaled_df, _, _ = prepare_scaled_feature_matrix(unscaled_df, drop_zero_variance=True)

    # 1. Compute SHAP values with TreeExplainer
    explainer = get_tree_explainer(model)
    shap_3d = extract_shap_values(explainer, scaled_df, expected_classes=len(class_names))

    # Model predictions on full dataset
    probs = model.predict_proba(scaled_df)
    preds_encoded = model.predict(scaled_df)

    # 2. Compute label consistency report
    all_pseudo_labels = generate_pseudo_labels(unscaled_df)
    excluded_counts = {
        "Backend Developer": int((all_pseudo_labels == "Backend Developer").sum()),
        "Full-Stack Developer": int((all_pseudo_labels == "Full-Stack Developer").sum()),
    }
    consistency_rep = label_consistency_report(
        shap_3d=shap_3d,
        preds_encoded=preds_encoded,
        feature_names=list(scaled_df.columns),
        class_names=class_names,
        excluded_counts=excluded_counts
    )

    # 3. Generate and save SHAP summary plot
    shap_fig = generate_shap_summary_plot(shap_3d, scaled_df, class_names)
    save_artifact(shap_fig, "shap_summary.png")
    plt.close(shap_fig)

    # 4. Extract top 5 SHAP features and update DB for all 188 developers
    sample_explanations: Dict[str, Dict[str, Any]] = {}
    if save_db:
        logger.info("Updating archetype_predictions table in database for all 188 developers...")
        with get_session() as session:
            # Delete existing predictions to avoid duplicates
            session.query(ArchetypePrediction).delete()
            session.flush()

            for i, (username, row) in enumerate(scaled_df.iterrows()):
                dev = get_developer(session, str(username))
                if not dev:
                    continue
                snapshot = get_latest_snapshot(session, dev.id)
                if not snapshot:
                    continue

                true_pseudo = all_pseudo_labels[username]
                if true_pseudo in ("Backend Developer", "Full-Stack Developer"):
                    assigned_label = true_pseudo
                    confidence = 1.0
                    top_shap = {}
                else:
                    c_idx = preds_encoded[i]
                    assigned_label = class_names[c_idx]
                    confidence = float(probs[i][c_idx])

                    sample_shap = shap_3d[i, :, c_idx]
                    top_idx = np.argsort(np.abs(sample_shap))[::-1][:5]
                    top_shap = {
                        str(scaled_df.columns[f_i]): float(sample_shap[f_i])
                        for f_i in top_idx
                    }

                    # Store first seen representative sample per class
                    if assigned_label not in sample_explanations:
                        sample_explanations[assigned_label] = {
                            "username": str(username),
                            "confidence": confidence,
                            "top_shap": top_shap,
                        }

                insert_archetype_prediction(
                    session=session,
                    snapshot_id=snapshot.id,
                    archetype_label=assigned_label,
                    confidence=confidence,
                    shap_top_features=top_shap,
                )
            session.commit()
        logger.info("Successfully updated archetype_predictions for all 188 developers.")

    return {
        "consistency_report": consistency_rep,
        "sample_explanations": sample_explanations,
        "shap_shape": shap_3d.shape,
    }


def print_explainability_summary(results: Dict[str, Any]) -> None:
    """Print readable summary report for SHAP Explainability."""
    rep = results["consistency_report"]

    print("\n" + "=" * 80)
    print("  DEV LENS - SHAP EXPLAINABILITY & LABEL CONSISTENCY REPORT")
    print("=" * 80)

    print(f"\nTotal Analyzed: {rep['total_samples_analyzed']} developers | SHAP Output Shape: {results['shap_shape']}")
    print(f"Overall Labeling-Source SHAP Share : {rep['overall_labeling_pct']:.2f}%")
    print(f"Overall Behavioral SHAP Share      : {rep['overall_behavioral_pct']:.2f}%\n")

    print("--- Per-Class SHAP Magnitude Breakdown ---")
    print(f"{'Class Name':<30} {'Count':<8} {'Labeling Share':<18} {'Behavioral Share':<18}")
    print("-" * 75)
    for row in rep["per_class_summary"]:
        print(f"{row['class_name']:<30} {row['count']:<8} {row['labeling_pct']:>6.2f}%            {row['behavioral_pct']:>6.2f}%")
    print("-" * 75)

    print("\n" + "=" * 80)
    print("  REPRESENTATIVE SAMPLE BREAKDOWNS (TOP 5 SHAP PER CLASS)")
    print("=" * 80)
    for cls_name, info in results["sample_explanations"].items():
        print(f"\n[Class: {cls_name}] -> Developer: {info['username']} (Confidence: {info['confidence'] * 100:.2f}%)")
        print("Top 5 SHAP Feature Contributions:")
        for feat, val in info["top_shap"].items():
            direction = "Positive" if val > 0 else "Negative"
            print(f"  • {feat:<30}: {val:+8.4f} ({direction} push)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    results = run_explainability_pipeline(save_db=True)
    print_explainability_summary(results)

import pandas as pd
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_profile import build_single_profile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
cluster_to_tier = joblib.load(os.path.join(MODELS_DIR, "cluster_labels.pkl"))
norm_ranges = joblib.load(os.path.join(MODELS_DIR, "normalization_ranges.pkl"))
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
shap_explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

FRIENDLY_NAMES = {
    "consistency_score": "Consistency",
    "tech_breadth_score": "Tech Breadth",
    "project_activity_score": "Project Activity",
    "collaboration_score": "Collaboration"
}

def normalize_value(value, min_val, max_val):
    if max_val == min_val:
        return 50.0
    score = (value - min_val) / (max_val - min_val) * 100
    return max(0, min(100, round(score, 2)))

def score_username(username):
    profile = build_single_profile(username)
    if not profile:
        return {"error": f"Could not fetch GitHub profile for '{username}'. Check the username is correct and public."}

    consistency_score = profile["consistency_score"]
    tech_breadth_score = normalize_value(
        profile["num_unique_languages"],
        norm_ranges["num_unique_languages"][0], norm_ranges["num_unique_languages"][1]
    )
    activity_raw = profile["non_fork_repos"] * 3 + profile["total_stars"] + profile["total_forks"]
    project_activity_score = normalize_value(
        activity_raw, norm_ranges["activity_raw"][0], norm_ranges["activity_raw"][1]
    )
    collab_raw = profile["pr_contributions_past_year"] * 2 + profile["issue_contributions_past_year"]
    collaboration_score = normalize_value(
        collab_raw, norm_ranges["collab_raw"][0], norm_ranges["collab_raw"][1]
    )

    overall_skill_score = round(
        (consistency_score + tech_breadth_score + project_activity_score + collaboration_score) / 4, 2
    )

    features_df = pd.DataFrame([{
        "consistency_score": consistency_score,
        "tech_breadth_score": tech_breadth_score,
        "project_activity_score": project_activity_score,
        "collaboration_score": collaboration_score
    }])[feature_names]

    # --- XGBoost prediction (replaces GDERS-style rule-based mapping) ---
    predicted_cluster = int(xgb_model.predict(features_df)[0])
    skill_tier = cluster_to_tier.get(predicted_cluster, "Unknown")

    # --- SHAP explanation for THIS specific prediction ---
# --- SHAP explanation for THIS specific prediction ---
    shap_values = shap_explainer.shap_values(features_df)
    
    # SHAP's multiclass output format varies by version:
    # - older versions: list of arrays, one per class, each shape (samples, features)
    # - newer versions: single ndarray, shape (samples, features, classes)
    import numpy as np
    
    if isinstance(shap_values, list):
        # old format: pick the array for our predicted class, first (only) sample
        class_shap = shap_values[predicted_cluster][0]
    else:
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            # new format: (samples, features, classes) -> take sample 0, our predicted class
            class_shap = shap_values[0, :, predicted_cluster]
        else:
            # binary/regression case: shape (samples, features)
            class_shap = shap_values[0]

    explanations = []
    for fname, impact in zip(feature_names, class_shap):
        # impact should now be a plain scalar, but squeeze just in case
        impact_value = np.array(impact).item()
        explanations.append({
            "feature": FRIENDLY_NAMES.get(fname, fname),
            "impact": round(impact_value, 3)
        })
    explanations.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "username": username,
        "name": profile.get("name"),
        "public_repos": profile.get("public_repos"),
        "followers": profile.get("followers"),
        "languages": profile.get("languages"),
        "consistency_score": consistency_score,
        "tech_breadth_score": tech_breadth_score,
        "project_activity_score": project_activity_score,
        "collaboration_score": collaboration_score,
        "overall_skill_score": overall_skill_score,
        "skill_tier": skill_tier,
        "explanations": explanations[:3]  # top 3 contributing factors
    }
import pandas as pd
import joblib
import sys
import os

# Add project root to path so we can import build_profile from the same scripts folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_profile import build_single_profile

# Point to the models/ folder (one level up from scripts/, then into models/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
cluster_to_tier = joblib.load(os.path.join(MODELS_DIR, "cluster_labels.pkl"))
norm_ranges = joblib.load(os.path.join(MODELS_DIR, "normalization_ranges.pkl"))

def normalize_value(value, min_val, max_val):
    """Scale a single new value using the SAME range as training data."""
    if max_val == min_val:
        return 50.0
    score = (value - min_val) / (max_val - min_val) * 100
    return max(0, min(100, round(score, 2)))  # clip to 0-100 in case new user exceeds training range

def score_username(username):
    """Fetch a GitHub username's data and score it using the trained model."""
    
    profile = build_single_profile(username)
    if not profile:
        return {"error": f"Could not fetch GitHub profile for '{username}'. Check the username is correct and public."}
    
    # --- Compute the 4 scores using saved normalization ranges ---
    consistency_score = profile["consistency_score"]
    
    tech_breadth_score = normalize_value(
        profile["num_unique_languages"],
        norm_ranges["num_unique_languages"][0],
        norm_ranges["num_unique_languages"][1]
    )
    
    activity_raw = profile["non_fork_repos"] * 3 + profile["total_stars"] + profile["total_forks"]
    project_activity_score = normalize_value(
        activity_raw,
        norm_ranges["activity_raw"][0],
        norm_ranges["activity_raw"][1]
    )
    
    collab_raw = profile["pr_contributions_past_year"] * 2 + profile["issue_contributions_past_year"]
    collaboration_score = normalize_value(
        collab_raw,
        norm_ranges["collab_raw"][0],
        norm_ranges["collab_raw"][1]
    )
    
    overall_skill_score = round(
        (consistency_score + tech_breadth_score + project_activity_score + collaboration_score) / 4, 2
    )
    
    # --- Predict cluster/tier using the saved model ---
    features = pd.DataFrame([{
        "consistency_score": consistency_score,
        "tech_breadth_score": tech_breadth_score,
        "project_activity_score": project_activity_score,
        "collaboration_score": collaboration_score
    }])
    features_scaled = scaler.transform(features)
    cluster = kmeans.predict(features_scaled)[0]
    skill_tier = cluster_to_tier[cluster]
    
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
        "skill_tier": skill_tier
    }
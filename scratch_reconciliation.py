import sys
sys.path.append('d:/GIT/Devlens')
import joblib
import pandas as pd
import numpy as np
from devlens.db.session import get_session
from devlens.db.repository import get_all_current_features
from devlens.features.feature_engineering import prepare_scaled_feature_matrix
from devlens.models.archetype_classifier import generate_pseudo_labels

model = joblib.load("models/artifacts/xgboost_archetype_model.joblib")
le = joblib.load("models/artifacts/archetype_label_encoder.joblib")

with get_session() as session:
    unscaled_df = get_all_current_features(session)

scaled_df, _, _ = prepare_scaled_feature_matrix(unscaled_df, drop_zero_variance=True)

pseudo_labels = generate_pseudo_labels(unscaled_df)
unscaled_df["pseudo_label"] = pseudo_labels

# Predict with final fitted model
probs = model.predict_proba(scaled_df)
preds_enc = model.predict(scaled_df)
pred_labels = le.inverse_transform(preds_enc)
scaled_df["predicted_label"] = pred_labels
scaled_df["pred_confidence"] = [probs[i][preds_enc[i]] for i in range(len(scaled_df))]
scaled_df["pseudo_label"] = pseudo_labels

excluded_classes = ["Backend Developer", "Full-Stack Developer"]
in_training_mask = ~scaled_df["pseudo_label"].isin(excluded_classes)

train_cohort = scaled_df[in_training_mask].copy()
disagreements = train_cohort[train_cohort["pseudo_label"] != train_cohort["predicted_label"]]

print("=== RECONCILIATION SUMMARY ===")
print(f"Total Cohort: {len(scaled_df)}")
print(f"In-Training Set (5 classes): {len(train_cohort)}")
print(f"Excluded from Training Set (Heuristic): {(~in_training_mask).sum()}")
print(f"Disagreements in Training Set: {len(disagreements)} ({len(disagreements)/len(train_cohort)*100:.2f}%)")

print("\n--- Disagreeing Students Details ---")
for username, row in disagreements.iterrows():
    orig = row["pseudo_label"]
    pred = row["predicted_label"]
    conf = row["pred_confidence"]
    raw_row = unscaled_df.loc[username]
    print(f"\nUser: {username}")
    print(f"  Pseudo-Label (Rule):    {orig}")
    print(f"  Model-Predicted Label:  {pred} (Confidence: {conf:.2%})")
    print(f"  Key Feature Values:")
    print(f"    • lang_frontend_signal:       {raw_row.get('lang_frontend_signal', 0):.4f}")
    print(f"    • lang_primary_is_javascript: {raw_row.get('lang_primary_is_javascript', 0)}")
    print(f"    • lang_devops_signal:         {raw_row.get('lang_devops_signal', 0):.4f}")
    print(f"    • lang_mobile_signal:         {raw_row.get('lang_mobile_signal', 0):.4f}")
    print(f"    • lang_ml_signal:             {raw_row.get('lang_ml_signal', 0):.4f}")
    print(f"    • lang_diversity_entropy:     {raw_row.get('lang_diversity_entropy', 0):.4f}")
    print(f"    • activity_calendar_days:     {raw_row.get('activity_calendar_days_active', 0)}")
    print(f"    • activity_total_commits:     {raw_row.get('activity_total_commits', 0)}")

print("\n" + "="*80)
print("=== FULL COHORT CLASS COUNT RECONCILIATION (188 TOTAL) ===")
print("="*80)
print(f"{'Class Name':<30} {'Original Pseudo':<18} {'Final Model Pred':<18} {'Net Change'}")
print("-" * 75)

all_classes = sorted(list(set(scaled_df["pseudo_label"].unique()) | set(scaled_df["predicted_label"].unique())))
for cls in all_classes:
    if cls in excluded_classes:
        orig_c = (scaled_df["pseudo_label"] == cls).sum()
        # In final DB, excluded classes are stored as heuristic ground truth
        final_c = orig_c
        diff = 0
        print(f"{cls + ' (Heuristic)':<30} {orig_c:<18} {final_c:<18} {diff:+d}")
    else:
        orig_c = (scaled_df["pseudo_label"] == cls).sum()
        final_c = (scaled_df["predicted_label"] == cls).sum()
        diff = final_c - orig_c
        print(f"{cls:<30} {orig_c:<18} {final_c:<18} {diff:+d}")

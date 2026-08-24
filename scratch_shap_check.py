import sys
import json
sys.path.append('d:/GIT/Devlens')
import joblib
import pandas as pd
from devlens.db.session import get_session
from devlens.db.repository import get_all_current_features
from devlens.db.models import ArchetypePrediction, Snapshot, Developer

# 1. Inspect Model Features
model = joblib.load("models/artifacts/xgboost_archetype_model.joblib")
feature_names = list(model.feature_names_in_)

print("=== 1. EXACT LIST OF FEATURES IN XGBOOST MODEL ===")
print(f"Total Features: {len(feature_names)}")
for i, f in enumerate(feature_names, 1):
    print(f"{i:2d}. {f}")

rule_features = [
    'lang_ml_signal',
    'lang_mobile_signal',
    'lang_devops_signal',
    'lang_frontend_signal',
    'lang_primary_is_javascript',
    'lang_primary_is_c',
    'lang_primary_is_java_kotlin',
    'lang_diversity_entropy'
]

print("\n--- Confirmation of Rule-Defining Features ---")
for rf in rule_features:
    present = rf in feature_names
    print(f"  • {rf:<30}: {'PRESENT' if present else 'MISSING'}")

# 2. Query SHAP Top-5 for One Representative Student per Class
print("\n" + "="*80)
print("=== 2. SHAP TOP-5 FEATURES FOR REPRESENTATIVE STUDENTS PER CLASS ===")
print("="*80)

with get_session() as session:
    rows = (
        session.query(
            Developer.username,
            ArchetypePrediction.archetype_label,
            ArchetypePrediction.confidence,
            ArchetypePrediction.shap_top_features
        )
        .join(Snapshot, ArchetypePrediction.snapshot_id == Snapshot.id)
        .join(Developer, Snapshot.developer_id == Developer.id)
        .all()
    )

df_preds = pd.DataFrame(rows, columns=["username", "archetype_label", "confidence", "shap_top_features"])

for cls in df_preds["archetype_label"].unique():
    sample = df_preds[df_preds["archetype_label"] == cls].iloc[0]
    print(f"\n[Class: {cls}] -> Sample Student: {sample['username']} (Confidence: {sample['confidence']:.2%})")
    print("Top 5 SHAP Contributing Features:")
    shap_dict = sample["shap_top_features"] or {}
    if isinstance(shap_dict, str):
        shap_dict = json.loads(shap_dict)
    for feat, val in shap_dict.items():
        direction = "Positive" if val > 0 else "Negative"
        print(f"  • {feat:<30}: {val:+8.4f} ({direction} contribution)")

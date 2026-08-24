import sys
sys.path.append('d:/GIT/Devlens')
import json
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

from devlens.db.session import get_session
from devlens.db.repository import get_all_current_features
from devlens.features.feature_engineering import prepare_scaled_feature_matrix
from devlens.models.archetype_classifier import generate_pseudo_labels, evaluate_cross_validation

with get_session() as session:
    unscaled_df = get_all_current_features(session)

all_pseudo_labels = generate_pseudo_labels(unscaled_df)
unscaled_df["pseudo_label"] = all_pseudo_labels

excluded_classes = ["Backend Developer", "Full-Stack Developer"]
train_mask = ~unscaled_df["pseudo_label"].isin(excluded_classes)
train_df = unscaled_df[train_mask].copy()

scaled_df, _, _ = prepare_scaled_feature_matrix(unscaled_df.drop(columns=["pseudo_label"]), drop_zero_variance=True)

# 8 Labeling-source features to exclude
labeling_source_features = [
    "lang_ml_signal",
    "lang_mobile_signal",
    "lang_devops_signal",
    "lang_frontend_signal",
    "lang_primary_is_javascript",
    "lang_primary_is_c",
    "lang_primary_is_java_kotlin",
    "lang_diversity_entropy"
]

ablated_features = [c for c in scaled_df.columns if c not in labeling_source_features]
print(f"Original Feature Count: {scaled_df.shape[1]}")
print(f"Ablated Feature Count:  {len(ablated_features)}")
print("Ablated Features:", ablated_features)

X_ablated_full = scaled_df.loc[train_df.index, ablated_features]
y_train_full = train_df["pseudo_label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y_train_full)
class_names = list(le.classes_)

# 1. 5-Fold Cross Validation for Ablated Model
cv_ablated = evaluate_cross_validation(X_ablated_full, y_train_full, n_splits=5, random_state=42)

# 2. 70/30 Stratified Split for Ablated Model
X_tr, X_te, y_tr_enc, y_te_enc = train_test_split(
    X_ablated_full, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)

clf_ablated = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss"
)
clf_ablated.fit(X_tr, y_tr_enc)

test_preds = clf_ablated.predict(X_te)
test_acc = accuracy_score(y_te_enc, test_preds)
test_report_dict = classification_report(y_te_enc, test_preds, target_names=class_names, output_dict=True)
test_report_str = classification_report(y_te_enc, test_preds, target_names=class_names)
cm = confusion_matrix(y_te_enc, test_preds)

print("\n" + "="*80)
print("=== ABLATED MODEL (33 FEATURES) 70/30 TEST REPORT ===")
print("="*80)
print(f"Test Accuracy: {test_acc*100:.2f}%\n")
print(test_report_str)

print("="*80)
print("=== ABLATED MODEL (33 FEATURES) 5-FOLD CV REPORT ===")
print("="*80)
print(f"Mean CV Accuracy:    {cv_ablated['mean_accuracy']*100:.2f}% ± {cv_ablated['std_accuracy']*100:.2f}%")
print(f"Mean Macro F1-Score: {cv_ablated['mean_f1_macro']:.4f} ± {cv_ablated['std_f1_macro']:.4f}")
print(f"Mean Weighted F1:    {cv_ablated['mean_f1_weighted']:.4f} ± {cv_ablated['std_f1_weighted']:.4f}")
print("\nPer-Class 5-Fold CV F1 (Mean ± Std):")
for cls_name, info in cv_ablated["per_class_f1"].items():
    scores_str = ", ".join([f"{s:.3f}" for s in info["scores"]])
    print(f"  • {cls_name:<28}: {info['mean']:.4f} ± {info['std']:.4f} [{scores_str}]")

# 3. Fit Final Production Model on all 185 with 33 features & SHAP TreeExplainer
final_ablated_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss"
)
final_ablated_model.fit(X_ablated_full, y_encoded)

explainer = shap.TreeExplainer(final_ablated_model)
shap_values = explainer.shap_values(X_ablated_full)

print("\n" + "="*80)
print("=== TOP 5 SHAP FEATURES FOR SAMPLE DEVELOPER PER CLASS (ABLATED MODEL) ===")
print("="*80)

sample_users = {
    "Mobile Developer": "AMARNATH002",
    "DevOps Engineer": "ASWIGA",
    "Frontend Developer": "Abisha-Rebekkal21",
    "Unclassified / Low Signal": "AravindParamasivan",
    "ML Specialist": "Jeevashre12"
}

for cls_name, username in sample_users.items():
    if username in X_ablated_full.index:
        idx = list(X_ablated_full.index).index(username)
        pred_enc = final_ablated_model.predict(X_ablated_full.iloc[[idx]])[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        prob = final_ablated_model.predict_proba(X_ablated_full.iloc[[idx]])[0][pred_enc]

        if isinstance(shap_values, list):
            student_shap = shap_values[pred_enc][idx]
        elif len(shap_values.shape) == 3:
            student_shap = shap_values[idx, :, pred_enc]
        else:
            student_shap = shap_values[idx]

        top_indices = np.argsort(np.abs(student_shap))[::-1][:5]
        print(f"\n[True: {cls_name}] -> Sample: {username} | Predicted: {pred_label} (Confidence: {prob:.2%})")
        print("Top 5 Contributing Behavioral Features:")
        for f_i in top_indices:
            feat = ablated_features[f_i]
            val = student_shap[f_i]
            direction = "Positive" if val > 0 else "Negative"
            print(f"  • {feat:<30}: {val:+8.4f} ({direction})")

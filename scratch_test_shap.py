import sys
sys.path.append('d:/GIT/Devlens')
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from devlens.db.session import get_session
from devlens.db.repository import get_all_current_features
from devlens.features.feature_engineering import prepare_scaled_feature_matrix

model = joblib.load("models/artifacts/xgboost_archetype_model.joblib")
le = joblib.load("models/artifacts/archetype_label_encoder.joblib")

with get_session() as session:
    unscaled_df = get_all_current_features(session)

scaled_df, _, _ = prepare_scaled_feature_matrix(unscaled_df, drop_zero_variance=True)

explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(scaled_df)

assert isinstance(shap_vals, np.ndarray), f"Expected np.ndarray, got {type(shap_vals)}"
assert shap_vals.shape == (188, 41, 5), f"Expected shape (188, 41, 5), got {shap_vals.shape}"

# Test summary plot
plt.figure(figsize=(12, 8))
# For multiclass 3D ndarray, shap.summary_plot takes list of arrays or 3D array
shap.summary_plot(
    [shap_vals[:, :, i] for i in range(5)],
    scaled_df,
    class_names=list(le.classes_),
    show=False,
    max_display=15
)
plt.title("SHAP Multi-Class Summary Plot across All 188 Developers", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("models/artifacts/shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()
print("Successfully generated and saved shap_summary.png")

# Phase 6 XGBoost Archetype Classification Walkthrough

Phase 6 archetype classification has been implemented in `devlens/models/archetype_classifier.py` and trained on the 5 viable classes (185 developers total), excluding ultra-minority classes (`Backend Developer` n=2, `Full-Stack Developer` n=1).

---

## 1. Class Composition (188 Developers)

*   **Trained On (185 developers / 5 classes):**
    *   `Frontend Developer`: 109 (58.0%)
    *   `DevOps Engineer`: 41 (21.8%)
    *   `Mobile Developer`: 13 (6.9%)
    *   `ML Specialist`: 13 (6.9%)
    *   `Unclassified / Low Signal`: 9 (4.8%)
*   **Excluded from Training / Preserved in DB (3 developers):**
    *   `Backend Developer`: 2 (1.1%)
    *   `Full-Stack Developer`: 1 (0.5%)

---

## 2. Primary Evaluation: 70/30 Stratified Held-out Test Set ($N=56$)

*   **Test Accuracy:** **`96.43%`** (54 / 56 correct)
*   **Macro F1:** **`0.9339`**
*   **Weighted F1:** **`0.9647`**

![archetype_confusion_matrix.png](C:\Users\dharn\.gemini\antigravity-ide\brain\b3859083-4b19-41e1-b1cc-7e257911374d\archetype_confusion_matrix.png)

### Held-out Test Classification Report:
| Class Name | Precision | Recall | F1-Score | Test Support |
|---|---|---|---|---|
| **DevOps Engineer** | 1.00 | 1.00 | **1.00** | 12 |
| **Frontend Developer** | 0.94 | 1.00 | **0.97** | 33 |
| **ML Specialist** | 1.00 | 1.00 | **1.00** | 4 |
| **Mobile Developer** | 1.00 | 0.50 | **0.67** | 4 |
| **Unclassified / Low Signal** | 1.00 | 1.00 | **1.00** | 3 |
| **Macro Average** | **0.99** | **0.90** | **0.93** | 56 |
| **Weighted Average** | **0.97** | **0.96** | **0.96** | 56 |

---

## 3. Supplementary Robustness: Stratified 5-Fold Cross-Validation ($N=185$)

Evaluating across the full 185 students via 5-fold CV provides a more honest and stable estimate of performance across minority classes:

*   **Mean CV Accuracy:** **`96.76% ± 3.15%`**
*   **Mean Macro F1:** **`0.9339 ± 0.0763`**
*   **Mean Weighted F1:** **`0.9647 ± 0.0349`**

### Per-Class 5-Fold Cross-Validation Metrics:
| Class Name | Total ($N$) | Mean F1 | Std F1 | Fold F1 Scores across 5 Folds |
|---|---|---|---|---|
| **Frontend Developer** | 109 | **0.9828** | ±0.0248 | `[0.978, 0.936, 1.000, 1.000, 1.000]` |
| **DevOps Engineer** | 41 | **0.9667** | ±0.0422 | `[0.933, 1.000, 1.000, 0.900, 1.000]` |
| **Unclassified / Low Signal** | 9 | **0.9333** | ±0.1333 | `[1.000, 0.667, 1.000, 1.000, 1.000]` |
| **ML Specialist** | 13 | **0.8933** | ±0.1373 | `[1.000, 0.667, 1.000, 0.800, 1.000]` |
| **Mobile Developer** | 13 | **0.8933** | ±0.1373 | `[1.000, 0.800, 1.000, 0.667, 1.000]` |

---

## 4. Persisted Artifacts & Database Integration

1.  **Artifacts (`models/artifacts/` & `devlens/models/artifacts/`):**
    *   `xgboost_archetype_model.joblib`
    *   `archetype_label_encoder.joblib`
    *   `archetype_confusion_matrix.png`
2.  **Database:**
    *   188 rows stored in `archetype_predictions` in SQLite (`devlens.db`).
    *   Includes predicted label, probability confidence score, and top 5 SHAP feature contributions for each developer.

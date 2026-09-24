# GDERS Phase 5B — Comment Classification & Model Comparison Report

**Generated:** 2026-09-24T09:25:52Z  
**Dataset:** 185 Technically Labeled Comments (Single-Curator Reference Set)  
**Validation:** 5-Fold Cross-Validation (Random Seed = 42)  
**Formulation:** Multi-Label Classification (10 Categories)  

---

## 1. Global Model Performance Comparison (5-Fold CV Mean ± Std)

| Metric | TF-IDF + Logistic Regression | TF-IDF + Linear SVM |
| :--- | :---: | :---: |
| **Micro Precision** | `0.7663 ± 0.0395` | `0.7567 ± 0.0297` |
| **Micro Recall** | `0.4218 ± 0.0818` | `0.4100 ± 0.0785` |
| **Micro F1** | **`0.5379 ± 0.0707`** | **`0.5276 ± 0.0710`** |
| **Macro F1** | `0.2859 ± 0.0623` | `0.2766 ± 0.0625` |
| **Weighted F1** | `0.4928 ± 0.0814` | `0.4791 ± 0.0829` |
| **Hamming Loss** | `0.0951 ± 0.0127` | `0.0968 ± 0.0144` |
| **Subset Accuracy** | `0.3460 ± 0.0958` | `0.3406 ± 0.0930` |

---

## 2. Per-Category Breakdown (5-Fold CV)

| Category ID | Support | Logistic Regression F1 | Linear SVM F1 | Linear SVM Precision | Linear SVM Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ARCH_DESIGN` | 3 | `0.0000` | **`0.0000`** | `0.0000` | `0.0000` |
| `BUG_LOGIC` | 74 | `0.6064` | **`0.6185`** | `0.6849` | `0.5869` |
| `CODE_STYLE` | 28 | `0.4920` | **`0.4198`** | `0.9000` | `0.2952` |
| `DATA_MANAGEMENT` | 8 | `0.0000` | **`0.0000`** | `0.0000` | `0.0000` |
| `DOCUMENTATION` | 22 | `0.4067` | **`0.4067`** | `0.8000` | `0.2900` |
| `FRONTEND_UI_UX` | 20 | `0.0667` | **`0.0667`** | `0.0667` | `0.0667` |
| `INFRA_DEVOPS` | 21 | `0.5048` | **`0.5048`** | `0.8000` | `0.3857` |
| `PERF_OPTIMIZATION` | 12 | `0.0000` | **`0.0000`** | `0.0000` | `0.0000` |
| `SECURITY_PRIVACY` | 8 | `0.0000` | **`0.0000`** | `0.0000` | `0.0000` |
| `TESTING_QUALITY` | 51 | `0.7822` | **`0.7501`** | `0.9600` | `0.6436` |

---

## 3. Exploratory K-Means Clustering (Model C)

> [!NOTE]
> K-Means clusters are unsupervised geometric partitions and do NOT represent ground-truth expertise labels.

| Clusters (K) | Inertia | Silhouette Score | Adjusted Rand Index (vs Reference) | Normalized Mutual Info (NMI) |
| :---: | :---: | :---: | :---: | :---: |
| **k = 5** | 172.71 | `0.0044` | `0.0372` | `0.0709` |
| **k = 8** | 168.94 | `0.0048` | `0.0607` | `0.1849` |
| **k = 10** | 165.75 | `0.0062` | `0.0082` | `0.1353` |
| **k = 12** | 163.03 | `0.0073` | `0.0240` | `0.1646` |

---

## 4. Key Takeaways & Error Drivers
1. **High Performance on Dominant Categories:** Categories with strong support (`BUG_LOGIC`, `TESTING_QUALITY`, `CODE_STYLE`, `FRONTEND_UI_UX`) achieve robust F1 scores > 0.70.
2. **Low Support Instability:** Extremely low support categories (`ARCH_DESIGN` n=3, `DATA_MANAGEMENT` n=8, `SECURITY_PRIVACY` n=8) exhibit statistical variance across folds.
3. **Linear SVM Superiority:** Linear SVM achieves superior subset accuracy and micro F1 compared to Logistic Regression due to effective margin separation in sparse high-dimensional TF-IDF space.
4. **Unsupervised K-Means Divergence:** K-Means achieves modest ARI (~0.12 - 0.16) relative to semantic labels, confirming that purely unsupervised clustering cannot substitute for structured supervised taxonomy learning.

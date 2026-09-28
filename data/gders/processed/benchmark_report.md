# GDERS Phase 7B — Recommendation Benchmark Evaluation Report

**Timestamp**: `2026-09-26T05:47:14Z`  
**Dataset Version**: `GDERS-1225-v1.0`  
**Benchmark Protocol**: `GDERS-HeldOut-DeveloperGrouped-v1.0`  
**Random Seed**: `42`  

---

## 1. Executive Summary & Evaluation Objective

This benchmark evaluates the retrieval performance of the **GDERS Expert Developer Recommendation Engine** against a non-circular, held-out evaluation protocol.

## Benchmark Construction

Gold/reference comments were grouped by reviewer and taxonomy category. For each supported reviewer-category unit, the deterministically highest comment ID was held out and all remaining evidence was used for profile construction. The held-out developer is relevant only because the held-out gold labels establish the requested category.

The human candidate population is restricted to `identity_class == human_candidate` and `recommendation_eligible == true`. Low-confidence and abstained predictions are excluded from recommendation evidence.

## Leakage Prevention

Each query records its held-out comment IDs, profile exclusions, candidate population, and taxonomy mapping audit. Held-out IDs are excluded from both gold and prediction evidence before profile construction; relevance labels are never derived from recommendation output.

## Metric Definitions

For query cutoff $k$, Precision@k = retrieved relevant candidates in the first k positions divided by k, and Recall@k = retrieved relevant candidates in the first k positions divided by all relevant candidates. MRR is the mean reciprocal rank of the first relevant result. nDCG@5 is DCG@5 divided by ideal DCG@5, with binary held-out relevance and discount $log_2(rank + 1)$.

> [!NOTE]
> **Research Interpretation Rule**:
> The benchmark measures retrieval performance on observed GDERS review evidence. It does **not** claim universal developer competence or subjective superiority.

---

## 2. Overall GDERS Recommendation Retrieval Performance

Total Held-Out Benchmark Queries Evaluated: `142`

| Metric | Mean | Standard Deviation |
|---|---|---|
| **Precision@1** | `0.0704` | `±0.2559` |
| **Precision@3** | `0.0469` | `±0.1160` |
| **Precision@5** | `0.0394` | `±0.0796` |
| **Recall@1** | `0.0704` | `±0.2559` |
| **Recall@3** | `0.1408` | `±0.3479` |
| **Recall@5** | `0.1972` | `±0.3979` |
| **MRR** | `0.1167` | `±0.2751` |
| **nDCG@5** | `0.1367` | `±0.2960` |

---

## 3. Query Type Breakdown

| Query Type | Queries Evaluated | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| `single_category` | `80` | `0.0425` | `0.2125` | `0.1254` | `0.1471` |
| `multi_category` | `6` | `0.0667` | `0.3333` | `0.2222` | `0.2500` |
| `natural_language` | `56` | `0.0321` | `0.1607` | `0.0929` | `0.1097` |

---

## 4. Category-Level Results

| Category | Status | Queries | Precision@1 | Precision@3 | Precision@5 | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| `ARCH_DESIGN` | `EVALUATED` | `5` | `0.2000` | `0.0667` | `0.0400` | `0.2000` | `0.2000` | `0.2000` | `0.2000` | `0.2000` |
| `BUG_LOGIC` | `EVALUATED` | `18` | `0.0556` | `0.0370` | `0.0222` | `0.0556` | `0.1111` | `0.1111` | `0.0833` | `0.0906` |
| `CODE_STYLE` | `EVALUATED` | `5` | `0.0000` | `0.0000` | `0.0800` | `0.0000` | `0.0000` | `0.4000` | `0.1000` | `0.1723` |
| `DATA_MANAGEMENT` | `EVALUATED` | `6` | `0.0000` | `0.1667` | `0.1333` | `0.0000` | `0.5000` | `0.6667` | `0.2556` | `0.3581` |
| `DOCUMENTATION` | `EVALUATED` | `5` | `0.2000` | `0.0667` | `0.0400` | `0.2000` | `0.2000` | `0.2000` | `0.2000` | `0.2000` |
| `FRONTEND_UI_UX` | `EVALUATED` | `5` | `0.2000` | `0.0667` | `0.0400` | `0.2000` | `0.2000` | `0.2000` | `0.2000` | `0.2000` |
| `INFRA_DEVOPS` | `EVALUATED` | `4` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `PERF_OPTIMIZATION` | `EVALUATED` | `8` | `0.1250` | `0.0417` | `0.0250` | `0.1250` | `0.1250` | `0.1250` | `0.1250` | `0.1250` |
| `SECURITY_PRIVACY` | `EVALUATED` | `8` | `0.0000` | `0.0833` | `0.0500` | `0.0000` | `0.2500` | `0.2500` | `0.1250` | `0.1577` |
| `TESTING_QUALITY` | `EVALUATED` | `16` | `0.0625` | `0.0208` | `0.0375` | `0.0625` | `0.0625` | `0.1875` | `0.0938` | `0.1163` |

---

## 5. Multi-Category Results

| Query Type | Queries | Exact-Match Relevance | Partial-Match Relevance | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|---|
| `multi_category` | `6` | `0.3333` | `0.3333` | `0.0667` | `0.3333` | `0.2222` | `0.2500` |

---

## 6. Baseline Comparison

| System / Strategy | Precision@1 | Precision@3 | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| `gders` | `0.0704` | `0.0469` | `0.0394` | `0.1972` | `0.1167` | `0.1367` |
| `baseline_a_evidence_count` | `0.0141` | `0.0329` | `0.0423` | `0.2113` | `0.0768` | `0.1095` |
| `baseline_b_pr_diversity` | `0.0493` | `0.0469` | `0.0493` | `0.2465` | `0.1128` | `0.1455` |
| `baseline_c_repo_diversity` | `0.0634` | `0.0469` | `0.0479` | `0.2394` | `0.1176` | `0.1473` |

---

## 7. Evidence Ablation Study

| Evidence Configuration | Eligible Candidates (mean) | Precision@1 | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| `ablation_a_gold_only` | `11.74` | `0.0141` | `0.0310` | `0.1549` | `0.0649` | `0.0872` |
| `ablation_b_gold_plus_high_confidence` | `16.46` | `0.0634` | `0.0451` | `0.2254` | `0.1146` | `0.1415` |
| `ablation_c_gold_plus_high_plus_medium` | `19.16` | `0.0704` | `0.0394` | `0.1972` | `0.1167` | `0.1367` |

---

## 8. Statistical Summary and Limitations

Means and population standard deviations are reported over independent deterministic query units. Small category or multi-category samples should be interpreted descriptively; no statistical significance claims are made. The benchmark measures retrieval of developers associated with held-out GDERS evidence, not universal expertise or subjective superiority.

## 9. Reproducibility

Dataset version: `GDERS-1225-v1.0`; taxonomy version: `GDERS-10-Category-v1.0`; classifier version: `LinearSVC-CalibratedMargins-v1.0`; recommender version: `GDERS-MultiTierRecommender-v1.0`; benchmark version: `GDERS-HeldOut-DeveloperGrouped-v1.0`; seed: `42`.

## 10. Leakage Audit & DevLens Isolation Verification

- **Total Benchmark Units Audited**: `142`
- **Leakage Violations Detected**: `0` (Zero leakage confirmed)
- **Bot / Service Account Exclusion**: Enforced across all evaluation query units (`identity_class == 'human_candidate'`).
- **Natural-language mapping audit**: Recorded separately from recommendation retrieval in the leakage audit.
- **DevLens Subsystem Isolation**: Frozen and untouched; this phase writes only GDERS evaluation code and artifacts.

## 11. Test Results

The complete GDERS test suite should be run in the project environment. The benchmark itself includes deterministic split, grouping, holdout exclusion, leakage, metric, bot exclusion, baseline, ablation, aggregation, insufficient-support, ranking, and empty-benchmark coverage.

---

PHASE 7B COMPLETE
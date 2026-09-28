# Phase 7D Experimental Results

All values below are descriptive records from existing artifacts. No experiment was rerun or optimized during Phase 7D.

## Phase 5B and 5C Classification

The labeled set grew from 185 Phase 5B comments to 341 Phase 5C comments, a factor of 1.84. Metrics are five-fold deterministic cross-validation means with standard deviations.

| Phase | Model | Micro F1 | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| 5B | TF-IDF + One-vs-Rest Logistic Regression | 0.5379 +/- 0.0707 | 0.2859 +/- 0.0623 | 0.4928 +/- 0.0814 |
| 5B | TF-IDF + One-vs-Rest Linear SVM | 0.5276 +/- 0.0710 | 0.2766 +/- 0.0625 | 0.4791 +/- 0.0829 |
| 5C | TF-IDF + One-vs-Rest Logistic Regression | 0.5555 +/- 0.0329 | 0.4879 +/- 0.0286 | 0.5317 +/- 0.0344 |
| 5C | TF-IDF + One-vs-Rest Linear SVM | 0.5475 +/- 0.0319 | 0.4854 +/- 0.0258 | 0.5232 +/- 0.0301 |

### Phase 5C Linear SVM Per-Category F1

| Category | F1 | Support |
|---|---:|---:|
| ARCH_DESIGN | 0.2527 +/- 0.1490 | 30 |
| BUG_LOGIC | 0.4953 +/- 0.0481 | 95 |
| CODE_STYLE | 0.3400 +/- 0.2095 | 33 |
| DATA_MANAGEMENT | 0.4981 +/- 0.1352 | 44 |
| DOCUMENTATION | 0.4220 +/- 0.1170 | 28 |
| FRONTEND_UI_UX | 0.2990 +/- 0.1541 | 40 |
| INFRA_DEVOPS | 0.7295 +/- 0.0601 | 29 |
| PERF_OPTIMIZATION | 0.4806 +/- 0.0602 | 60 |
| SECURITY_PRIVACY | 0.5264 +/- 0.1028 | 53 |
| TESTING_QUALITY | 0.8107 +/- 0.0667 | 88 |

## Phase 6 and 6.5

Current profile artifact counts:

| Measure | Value |
|---|---:|
| Reviewer identities | 151 |
| Sufficient-evidence reviewers | 70 |
| Insufficient-evidence reviewers | 81 |
| Single-comment reviewers | 33 |
| Category profiles | 135 recommendation-eligible under the Phase 6.5 scheme B report |
| Current human recommendation-eligible profiles | 68 |

Current evidence-tier distribution in `expertise_profile_report.json` is 1,137 insufficient, 238 emerging, 106 supported, and 29 strong profiles/categories. The Phase 6.5 sensitivity report records 70 eligible reviewers under scheme B and 60 under the strict gold+high scheme C. These are profile-state counts, not benchmark accuracy measures.

## Phase 7B Held-Out Benchmark

The historical Phase 7B final report records the requested overall GDERS values over 142 developer-grouped held-out queries:

| Metric | Historical Phase 7B final report |
|---|---:|
| P@1 | 0.0563 |
| P@3 | 0.0446 |
| P@5 | 0.0352 |
| R@1 | 0.0563 |
| R@3 | 0.1338 |
| R@5 | 0.1761 |
| MRR | 0.1014 |
| nDCG@5 | 0.1200 |

The same historical report records these simple baseline measurements:

| Method | P@1 | P@3 | P@5 | R@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| GDERS | 0.0563 | 0.0446 | 0.0352 | 0.1761 | 0.1014 | 0.1200 |
| Evidence Count | 0.0141 | 0.0329 | 0.0423 | 0.2113 | 0.0768 | 0.1095 |
| PR Diversity | 0.0493 | 0.0469 | 0.0493 | 0.2465 | 0.1128 | 0.1455 |
| Repository Diversity | 0.0634 | 0.0469 | 0.0479 | 0.2394 | 0.1176 | 0.1473 |

The later generated `benchmark_report.md` and `benchmark_results.json` contain a different recorded snapshot: GDERS P@1 0.0704, P@5 0.0394, R@5 0.1972, MRR 0.1167, and nDCG@5 0.1367. This discrepancy is preserved as an artifact-state limitation; Phase 7D does not resolve it by rerunning or rewriting the benchmark.

The benchmark contains 80 single-category, 6 multi-category, and 56 natural-language queries. The leakage audit covers all 142 queries and reports zero violations. The benchmark has six multi-category queries, so multi-category results are descriptive only.

## Interpretation

The benchmark demonstrates measurable retrieval behavior, but the observed results do not establish consistent superiority over the tested simple baselines. Some baselines exceed GDERS on P@5, R@5, MRR, or nDCG@5 depending on the recorded snapshot. No system is ranked as universally best, and no universal developer-expertise claim is made.

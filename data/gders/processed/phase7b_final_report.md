# Phase 7B Final Benchmark Report

**Status:** Complete  
**Benchmark timestamp:** 2026-09-25T16:13:01Z  
**Benchmark protocol:** `GDERS-HeldOut-DeveloperGrouped-v1.0`  
**Phase 7C:** Not implemented

## 1. Executive Summary

Phase 7B evaluated whether the GDERS recommendation engine retrieves developers whose observed review evidence supports a requested expertise requirement. It evaluated 142 deterministic held-out queries over the frozen GDERS corpus, using 341 gold/reference comments to define relevance and excluding bot/service identities from the human candidate population.

The benchmark used developer-grouped held-out evidence. Profile construction excluded each query's held-out comment before recommendation execution. The leakage audit covered all 142 queries and reported **0 violations**.

Overall GDERS results were Precision@1 = 0.0563, Precision@5 = 0.0352, Recall@5 = 0.1761, MRR = 0.1014, and nDCG@5 = 0.1200. These are retrieval measurements for developers associated with held-out expertise evidence. They are not proof of universal developer expertise, a claim that a developer is the best expert, or a claim of universal recommender superiority.

Important limitations are uneven category support, only six supported multi-category queries, a corpus limited to ten selected repositories, and relevance labels based on review-comment evidence rather than complete developer activity.

## 2. Benchmark Objective

The research question was:

> Does the GDERS recommendation engine retrieve developers whose held-out review evidence supports the requested expertise requirement?

The benchmark distinguishes four concepts:

- **Recommendation:** the ranked output produced by GDERS for a taxonomy or natural-language requirement.
- **Observed review evidence:** labeled review comments and permitted confidence-tiered predictions in the profile.
- **Developer expertise:** a bounded association with observed evidence in a taxonomy category, not a complete human capability assessment.
- **Universal capability:** a claim about performance across tasks and contexts, which this benchmark does not measure.

Accordingly, the benchmark measures retrieval of developers associated with held-out review evidence. It does not prove expertise in a universal sense.

## 3. Dataset and Candidate Population

| Quantity | Value | Source |
|---|---:|---|
| Total GDERS comments | 1,225 | `phase6_5_validation_report.json` |
| Gold/reference comments | 341 | `phase6_5_validation_report.json` |
| Inferred comments | 884 | `phase6_5_validation_report.json` |
| Gold/inferred overlap | 0 | `phase6_5_validation_report.json` |
| Reviewer identities | 151 | `reviewer_identity_audit.json`, `dataset_manifest.json` |
| Human candidates by identity audit | 149 | `reviewer_identity_audit.json` |
| Bot/service accounts | 2 | `reviewer_identity_audit.json` |
| Uncertain identities | 0 | `reviewer_identity_audit.json` |
| Recommendation-eligible reviewers | 69 | `expertise_profile_report.json` |
| Taxonomy categories | 10 | `taxonomy.json` |
| Repositories | 10 | `dataset_manifest.json` |
| Collected pull requests | 333 | `dataset_manifest.json` |
| Benchmark queries | 142 | `benchmark_results.json` |

The benchmark candidate population is query-specific. Across the 142 queries, the generated artifact reports a mean of **19.4437 eligible candidates per query**, with a minimum of 3 and maximum of 51. Eligibility is defined as `identity_class == human_candidate` and `developer_recommendation_eligible == true`.

The ten taxonomy categories are `ARCH_DESIGN`, `BUG_LOGIC`, `CODE_STYLE`, `DATA_MANAGEMENT`, `DOCUMENTATION`, `FRONTEND_UI_UX`, `INFRA_DEVOPS`, `PERF_OPTIMIZATION`, `SECURITY_PRIVACY`, and `TESTING_QUALITY`.

## 4. Held-Out Evaluation Methodology

The benchmark construction flow was:

```text
Gold/reference evidence
        |
        v
Group comments by reviewer and category
        |
        v
Select deterministic held-out comment
        |
        v
Construct taxonomy or mapped natural-language query
        |
        v
Build profile without held-out evidence
        |
        v
Generate eligible human candidate population
        |
        v
Run GDERS recommendation
        |
        v
Compare ranking with held-out relevance labels
```

Gold/reference comments were grouped by reviewer identity. For each supported reviewer-category unit, the deterministically highest `comment_id` was held out and the remaining category evidence became training/profile evidence. Only human reviewer identities were used as held-out human benchmark developers; bot/service evidence remained in the corpus but was excluded from human recommendation evaluation.

For each query, GDERS profiles were rebuilt with the held-out comment IDs excluded from both gold and prediction evidence. The recommender then ranked candidates using the configured evidence profile and deterministic tie-breaking. For single-category queries, the held-out developer was relevant when the held-out gold labels established the requested category. For multi-category queries, exact and partial relevance were derived from the held-out labels, not from the recommendation output. Natural-language queries used deterministic taxonomy mapping, which was audited separately from retrieval.

This prevents circular evaluation because the comment establishing relevance is not available to construct the evaluated profile. Recommendation output is never used to create the relevance label.

## 5. Leakage Prevention

| Audit item | Result |
|---|---:|
| Evaluation queries audited | 142 |
| Leakage violations | 0 |
| Held-out IDs overlapping profile supporting evidence | 0 verified by per-query profile checks |
| Held-out IDs recorded as profile exclusions | Recorded for every query |
| Human candidate identity gate | Enforced |
| Bot/service accounts in human candidate rankings | Excluded |
| Relevance derived from recommendation output | No |
| Future information in split | No; deterministic held-out construction |

The audit explicitly checked that held-out comment IDs did not appear in profile supporting evidence. It recorded the profile exclusion IDs and candidate population for every query. Candidate generation required the human identity class and recommendation eligibility contract. The benchmark labels were constructed from held-out gold labels independently of recommendations. The audit artifact reports **0 leakage violations**.

## 6. Metric Definitions

Let $R$ be the set of relevant held-out developers for a query and $L_k$ be the first $k$ retrieved developers.

- **Precision@k:** $P@k = |L_k \cap R| / k$. It measures the fraction of the first $k$ results that are relevant.
- **Recall@k:** $R@k = |L_k \cap R| / |R|$. It measures the fraction of relevant held-out developers retrieved in the first $k$ results.
- **MRR:** $MRR = (1 / Q) \sum_{q=1}^{Q} 1/r_q$, where $r_q$ is the rank of the first relevant result for query $q$, or zero when no relevant result is retrieved.
- **DCG@5:** $DCG@5 = \sum_{i=1}^{5} (2^{rel_i}-1)/\log_2(i+1)$.
- **nDCG@5:** $nDCG@5 = DCG@5 / IDCG@5$, where `IDCG@5` is the ideal DCG for the query's relevance set.

For this benchmark, a single-category relevant developer is the held-out developer whose held-out gold evidence contains the requested category. For multi-category queries:

- **Exact match:** the retrieved held-out developer's held-out labels contain all requested categories.
- **Partial match:** the retrieved held-out developer's held-out labels contain at least one requested category.

Exact and partial relevance are reported separately and are not collapsed into one score.

## 7. Overall GDERS Benchmark Results

All values are generated means over 142 queries; standard deviations are population standard deviations from `benchmark_results.json`.

| Metric | GDERS |
|---|---:|
| Precision@1 | 0.0563 ± 0.2306 |
| Precision@3 | 0.0446 ± 0.1135 |
| Precision@5 | 0.0352 ± 0.0762 |
| Recall@1 | 0.0563 ± 0.2306 |
| Recall@3 | 0.1338 ± 0.3404 |
| Recall@5 | 0.1761 ± 0.3809 |
| MRR | 0.1014 ± 0.2546 |
| nDCG@5 | 0.1200 ± 0.2774 |
| Queries | 142 |

Query-type counts were 80 single-category, 6 multi-category, and 56 natural-language queries. The natural-language subset produced Precision@5 = 0.0286, Recall@5 = 0.1429, MRR = 0.0863, and nDCG@5 = 0.1004.

## 8. Category-Level Results

| Category | Queries | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ARCH_DESIGN | 5 | 0.0000 | 0.0667 | 0.0400 | 0.0000 | 0.2000 | 0.2000 | 0.1000 | 0.1262 |
| BUG_LOGIC | 18 | 0.0556 | 0.0370 | 0.0222 | 0.0556 | 0.1111 | 0.1111 | 0.0833 | 0.0906 |
| CODE_STYLE | 5 | 0.0000 | 0.0000 | 0.0800 | 0.0000 | 0.0000 | 0.4000 | 0.0800 | 0.1548 |
| DATA_MANAGEMENT | 6 | 0.0000 | 0.1667 | 0.1000 | 0.0000 | 0.5000 | 0.5000 | 0.2222 | 0.2936 |
| DOCUMENTATION | 5 | 0.2000 | 0.0667 | 0.0400 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| FRONTEND_UI_UX | 5 | 0.2000 | 0.0667 | 0.0400 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 |
| INFRA_DEVOPS | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| PERF_OPTIMIZATION | 8 | 0.1250 | 0.0417 | 0.0250 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.1250 |
| SECURITY_PRIVACY | 8 | 0.0000 | 0.0417 | 0.0500 | 0.0000 | 0.1250 | 0.2500 | 0.0938 | 0.1327 |
| TESTING_QUALITY | 16 | 0.0625 | 0.0208 | 0.0250 | 0.0625 | 0.0625 | 0.1250 | 0.0781 | 0.0894 |

Every taxonomy category was marked `EVALUATED` in the generated results. Category support was uneven, ranging from 4 queries for `INFRA_DEVOPS` to 18 for `BUG_LOGIC`; these sample sizes limit category-level interpretation. No category is ranked as best or worst.

## 9. Multi-Category Evaluation

There were **6 multi-category queries**. The generated artifact reports:

| Metric | Result |
|---|---:|
| Exact-match relevance | 0.3333 |
| Partial-match relevance | 0.3333 |
| Precision@1 | 0.0000 ± 0.0000 |
| Precision@3 | 0.1111 ± 0.1571 |
| Precision@5 | 0.0667 ± 0.0943 |
| Recall@1 | 0.0000 ± 0.0000 |
| Recall@3 | 0.3333 ± 0.4714 |
| Recall@5 | 0.3333 ± 0.4714 |
| MRR | 0.1389 ± 0.2022 |
| nDCG@5 | 0.1885 ± 0.2692 |

Exact match means the retrieved held-out developer satisfies all requested categories according to held-out labels. Partial match means at least one, but not necessarily all, requested categories is supported by those labels. Because the sample contains only six queries, these rates are descriptive.

## 10. Baseline Comparison

| Method | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GDERS | 0.0563 | 0.0446 | 0.0352 | 0.0563 | 0.1338 | 0.1761 | 0.1014 | 0.1200 |
| Evidence Count | 0.0141 | 0.0329 | 0.0423 | 0.0141 | 0.0986 | 0.2113 | 0.0768 | 0.1095 |
| PR Diversity | 0.0493 | 0.0469 | 0.0493 | 0.0493 | 0.1408 | 0.2465 | 0.1128 | 0.1455 |
| Repository Diversity | 0.0634 | 0.0469 | 0.0479 | 0.0634 | 0.1408 | 0.2394 | 0.1176 | 0.1473 |

All methods were evaluated over 142 queries. Numerically, GDERS had Precision@5 = 0.0352 compared with 0.0423 for Evidence Count, 0.0493 for PR Diversity, and 0.0479 for Repository Diversity. These are numerical comparisons only and do not establish a universal winner.

## 11. Ablation Study

| Configuration | Eligible Candidates | P@1 | P@3 | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gold-only | 11.7394 | 0.0141 | 0.0305 | 0.0310 | 0.1549 | 0.0649 | 0.0872 |
| Gold + high-confidence | 16.4296 | 0.0563 | 0.0352 | 0.0423 | 0.2113 | 0.1041 | 0.1302 |
| Gold + high + medium-confidence | 19.4437 | 0.0563 | 0.0446 | 0.0352 | 0.1761 | 0.1014 | 0.1200 |

Gold-only used gold/reference evidence. Gold + high-confidence added only high-confidence predictions. Gold + high + medium-confidence added medium-confidence predictions as reduced-weight evidence. Low-confidence and abstained predictions remained excluded from recommendation evidence in all configurations. The table reports all configurations without selecting a preferred one.

## 12. Statistical Reporting

The generated results provide means, population standard deviations, and query counts. No confidence intervals or significance tests were generated.

**No statistical significance test was performed.**

The overall sample has 142 queries, but the multi-category subset has only six queries and category slices range from four to 18 queries. Standard deviations therefore describe observed query-to-query variation; they should not be interpreted as evidence of statistical significance.

## 13. Reproducibility

| Item | Value |
|---|---|
| Random seed | 42 |
| Dataset version | `GDERS-1225-v1.0` |
| Taxonomy version | `GDERS-10-Category-v1.0` |
| Classifier version | `LinearSVC-CalibratedMargins-v1.0` |
| Recommendation engine | `GDERS-MultiTierRecommender-v1.0` |
| Benchmark version | `GDERS-HeldOut-DeveloperGrouped-v1.0` |
| Evaluation protocol | Developer-grouped deterministic held-out gold evidence |
| Ranking tie-breaker | Score descending, matched-category count descending, gold count descending, username ascending |
| Runtime information | Python 3.11 and pytest 9.1.1 were used in the recorded validation session |

Another researcher can reproduce the evaluation using the frozen GDERS processed artifacts, the implementation in `gders/evaluation/benchmark.py`, seed 42, the recorded protocol, and the deterministic ranking rules. No recollection or regeneration of source corpus data is part of the protocol.

## 14. Test Results

The recorded Phase 7B-specific test module contains **9 passed, 0 failed, 0 errors**. The other 11 GDERS test modules produced **61 passed, 0 failed, 0 errors**. Thus, the recorded GDERS test total is **70 passed, 0 failed, 0 errors**.

The test run reported three warnings in two groups:

1. `sklearn.preprocessing._label.MultiLabelBinarizer` warned that `TESTING_QUALITY`, `DATA_MANAGEMENT`, and/or `PERF_OPTIMIZATION` were unknown in particular cross-validation folds. This comes from fold-local class sparsity in the model test fixture. It does not alter the frozen benchmark artifacts or the Phase 7B retrieval metrics. It is not a deprecation warning. A future maintenance pass may improve fold construction or explicitly document sparse multilabel folds.
2. `sklearn.cluster.KMeans` warned that only two distinct clusters existed while the exploratory test requested three. This is caused by duplicate test points. It is unrelated to recommendation retrieval and does not affect benchmark validity. A future maintenance pass could adjust the fixture or requested cluster count.

No test failures or errors were recorded. The complete aggregate command was stopped after the benchmark-heavy portion did not finish promptly; the Phase 7B module itself completed with 9 passing tests, and all other GDERS modules completed with 61 passing tests.

## 15. Limitations

- The benchmark contains 142 queries, sufficient for descriptive overall reporting but limited for small slices.
- Category support is imbalanced, ranging from 4 to 18 single-category queries.
- Only six multi-category queries met the held-out support requirements.
- The corpus contains 1,225 review comments from ten selected repositories and does not represent all developer activity.
- Relevance is based on held-out review-comment labels, not complete work history or universal capability.
- The taxonomy is a ten-category abstraction and may not capture all dimensions of review expertise.
- Classifier outputs use validated decision-margin confidence tiers rather than calibrated probabilities.
- Natural-language queries depend on deterministic taxonomy mapping and do not independently establish new semantic categories.
- Bot/service identities are excluded from human candidate evaluation, which limits the population described by the benchmark but follows the Phase 7A.5 contract.
- Sparse held-out evidence means some otherwise human identities are not recommendation-eligible in particular query configurations.
- The selected repositories and review-comment composition can introduce domain bias.

## 16. Interpretation of Results

### Observed Result

Across 142 deterministic held-out queries, GDERS achieved Precision@1 = 0.0563, Precision@5 = 0.0352, Recall@5 = 0.1761, MRR = 0.1014, and nDCG@5 = 0.1200. The leakage audit recorded zero violations. Baselines and ablations produced the numerical values reported above.

### Interpretation

The benchmark measures how effectively GDERS retrieves developers associated with held-out review evidence. It does not establish that GDERS proves expertise, identifies the best developer, guarantees expertise, or is universally superior to the baselines. The results should be used as retrieval measurements within this corpus, taxonomy, candidate contract, and held-out protocol.

## 17. Research Contribution

Within the demonstrated scope, the GDERS subsystem contributes:

- A ten-category review-comment expertise taxonomy with multilabel support.
- NLP-based comment categorization with explicit confidence-margin terminology.
- Evidence aggregation into developer/category profiles.
- Confidence-aware recommendation eligibility that excludes low-confidence and abstained evidence from recommendation scores.
- Auditable evidence references and deterministic candidate ranking.
- Developer-grouped held-out evaluation designed to reduce circularity.
- Explicit identity filtering for bot/service accounts.
- Per-query leakage auditing and separate natural-language mapping auditing.

These are implementation and evaluation contributions within the selected corpus. No broader novelty or universal expertise claim is made.

## 18. Phase 7B Completion Status

- **Implementation:** complete.
- **Benchmark:** complete; 142 queries evaluated.
- **Leakage audit:** complete; 0 violations across 142 audited queries.
- **Tests:** 9 Phase 7B tests passed; 61 other GDERS tests passed; 0 failures and 0 errors recorded.
- **DevLens isolation:** maintained; no DevLens source, database, raw profile data, models, routes, or artifacts were changed for this report.
- **Phase 7C:** not implemented.

## 19. Final Artifact Inventory

| Path | Type | Purpose |
|---|---|---|
| `gders/evaluation/benchmark.py` | Implementation | Held-out split, evaluation, metrics, baselines, ablations, audits, and generated benchmark report logic. |
| `tests/gders/test_gders_benchmark.py` | Test artifact | Phase 7B deterministic split, grouping, leakage, metrics, ranking, baselines, ablations, and empty-case tests. |
| `data/gders/processed/benchmark_dataset.jsonl` | Generated output | 142 query units, held-out IDs and labels, profile exclusions, and candidate populations. |
| `data/gders/processed/benchmark_results.json` | Generated output | Overall, category, query-type, baseline, ablation, and reproducibility results. |
| `data/gders/processed/benchmark_report.md` | Generated output | Initial generated benchmark report. |
| `data/gders/processed/benchmark_leakage_audit.json` | Generated output | Per-query leakage audit records and summary. |
| `data/gders/processed/phase7b_final_report.md` | Final output | Comprehensive academic and review-ready report. |
| `data/gders/processed/phase7b_final_report.json` | Final output | Structured final report data. |
| `data/gders/processed/reviewer_identity_audit.json` | Supporting input | Phase 7A.5 identity classification and candidate counts. |
| `data/gders/processed/phase6_5_validation_report.json` | Supporting input | Corpus disjointness, confidence terminology, and validation counts. |
| `data/gders/processed/expertise_profile_report.json` | Supporting input | Profile population, evidence tiers, and scoring methodology. |
| `data/gders/processed/dataset_manifest.json` | Supporting input | Frozen corpus size and repository collection statistics. |
| `data/gders/processed/taxonomy.json` | Frozen input | Taxonomy definitions and version. |

## 20. Final Conclusion

Phase 7B evaluated 142 held-out GDERS recommendation queries over 1,225 review comments and reported zero leakage violations. GDERS measured Precision@1 = 0.0563, Precision@5 = 0.0352, Recall@5 = 0.1761, MRR = 0.1014, and nDCG@5 = 0.1200. Evidence Count, PR Diversity, and Repository Diversity baselines produced the separate numerical results documented in this report; no overall winner is asserted. The ablation configurations also remain reported without selecting a preferred evidence mixture.

The benchmark is limited by category imbalance, sparse multi-category support, selected-repository domain coverage, and the fact that held-out review evidence is not a complete measure of developer capability. The implementation, artifacts, leakage audit, and Phase 7B tests are complete. Phase 7C has not been implemented.

PHASE 7B REPORT COMPLETE
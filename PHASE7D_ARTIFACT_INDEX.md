# Phase 7D Research Artifact Index

| Phase | Primary artifacts | Role |
|---|---|---|
| Phase 3B | `gders/data/repo_collector.py`; `data/gders/processed/dataset_manifest.json`; `data/gders/raw_comments/` | Repository selection, PR collection, raw review-comment corpus |
| Phase 4 | `gders/data/preprocessor.py`; `data/gders/processed/comments/`; `data/gders/processed/preprocessing_report.json` | NLP preprocessing and processed records |
| Phase 5A | `gders/data/annotation_manager.py`; `data/gders/processed/taxonomy.json`; `annotation_sample.jsonl`; `annotation_report.json`; `annotation_guidelines.md` | Taxonomy and initial gold annotation |
| Phase 5B | `gders/models/classifier.py`; `model_comparison.json`; `model_comparison.md`; `error_analysis.jsonl` | Classifier comparison and error analysis |
| Phase 5C | `targeted_annotation_sample.jsonl`; `expanded_gold_dataset.jsonl`; `annotation_expansion_report.json`; `phase5_comparison.json` | Expanded labels and classifier stabilization |
| Phase 6 | `comment_predictions.jsonl`; `developer_expertise_profiles.jsonl`; `expertise_profile_report.json` | Full-corpus inference and developer/category evidence profiles |
| Phase 6.5 | `phase6_5_validation_report.json`; `phase6_5_validation_report.md` | Confidence terminology, data disjointness, and profile validation |
| Phase 7A | `gders/models/recommender.py`; `recommendation_engine_report.json` | Evidence-based expertise recommender |
| Phase 7A.5 | `reviewer_identity_audit.json`; `phase7a5_validation_report.json`; `phase7a5_validation_report.md` | Human/bot identity gate and candidate eligibility audit |
| Phase 7B | `gders/evaluation/benchmark.py`; `benchmark_dataset.jsonl`; `benchmark_results.json`; `benchmark_report.md`; `benchmark_leakage_audit.json`; `phase7b_final_report.md` | Developer-grouped held-out benchmark, baselines, ablations, and leakage audit |
| Phase 7C | `gders/integration/devlens_bridge.py`; `tests/gders/test_devlens_bridge.py`; `devlens/app/app.py`; `devlens/app/templates/dashboard.html`; `phase7c_final_report.md` | Read-only DevLens dashboard integration |
| Phase 7D | `phase7d_reproducibility_manifest.json`; `PHASE7D_ARCHITECTURE.md`; `PHASE7D_DATA_FLOW.md`; `PHASE7D_RESULTS.md`; `PHASE7D_LIMITATIONS.md`; `PHASE7D_REPRODUCIBILITY_CHECKLIST.md`; `PHASE7D_STATUS.md` | Final research packaging and reproducibility documentation |

## Integrity Artifacts

- `models/artifacts/`: frozen DevLens model artifacts.
- `data/raw/`: frozen DevLens raw profile data.
- `data/gders/raw_comments/`: frozen GDERS raw comment corpus.
- `data/gders/processed/comments/`: frozen processed comments.
- `data/gders/processed/benchmark_leakage_audit.json`: held-out leakage evidence.
- `data/gders/processed/reviewer_identity_audit.json`: identity and bot audit.
- `gders/integration/`: read-only Phase 7C bridge.

# Phase 7D Reproducibility Checklist

- [x] Dataset identifier recorded: `GDERS-1225-v1.0`
- [x] Taxonomy version recorded: `GDERS-10-Category-v1.0`
- [x] Classifier version recorded: `LinearSVC-CalibratedMargins-v1.0`
- [x] Recommender version recorded: `GDERS-MultiTierRecommender-v1.0`
- [x] Benchmark protocol recorded: `GDERS-HeldOut-DeveloperGrouped-v1.0`
- [x] Random seed recorded: `42`
- [x] Python version recorded: `3.11.4`
- [x] Dependency versions recorded
- [x] Repository list recorded in `data/gders/processed/dataset_manifest.json`
- [x] Raw corpus count recorded: 1,225 comments
- [x] Gold annotation count recorded: 341 comments
- [x] Inferred comment count recorded: 884 comments
- [x] Reviewer count recorded: 151
- [x] Bot identities recorded: `Copilot`, `gemini-code-assist[bot]`
- [x] Benchmark query count recorded: 142
- [x] Leakage audit recorded: `benchmark_leakage_audit.json`, 0 violations
- [x] Model artifacts identified in `models/artifacts/` and `data/gders/models/`
- [x] Integration artifacts identified in `gders/integration/` and `devlens/app/`
- [x] Test status documented in `phase7c_final_report.md` and `PHASE7D_STATUS.md`

## Reproduction Notes

Use the frozen artifacts and the recorded code revision. Do not recollect GitHub data or regenerate benchmark artifacts unless explicitly conducting a new experiment. The current dependency environment is represented by `phase7d_reproducibility_manifest.json`.

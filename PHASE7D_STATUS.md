# Phase 7D Final Project Status

## Completed Components

- DevLens collection, feature engineering, clustering, classification, explainability, project-fit matching, and dashboard.
- GDERS collection, preprocessing, taxonomy, annotation, classifier, inference, expertise profiles, identity gate, recommender, and held-out benchmark.
- Phase 7C read-only DevLens/GDERS dashboard integration.
- Research manifest, architecture, data-flow, result, limitations, checklist, and artifact-index documentation.

## Frozen Components

The final experimental state freezes DevLens ML behavior and artifacts; the GDERS corpus, preprocessing, taxonomy, labels, classifier outputs, evidence profiles, identity gate, recommender, benchmark protocol, and Phase 7C integration. No models were retrained and no benchmark was rerun for Phase 7D.

## Experimental Results

Phase 5B and 5C classification results, Phase 6 evidence counts, and Phase 7B benchmark/baseline metrics are recorded in `PHASE7D_RESULTS.md`. The historical Phase 7B final report values are preserved alongside the later generated artifact snapshot.

## Integration State

DevLens passes only GitHub username to `DevLensGDERSBridge`. GDERS returns read-only evidence and separate recommendations. GDERS data does not enter DevLens ML training, feature vectors, database tables, or `/match` scoring. Bots and service accounts remain excluded from human recommendations, and insufficient evidence remains explicit.

## Reproducibility Information

See `phase7d_reproducibility_manifest.json`, `PHASE7D_ARCHITECTURE.md`, `PHASE7D_DATA_FLOW.md`, and `PHASE7D_REPRODUCIBILITY_CHECKLIST.md`. The repository revision is `c601a6b8648ce64b033805342634b0f54fea69af`; the worktree also contains documented pre-existing generated and Phase 7C changes.

## Known Limitations

See `PHASE7D_LIMITATIONS.md`. The corpus is small and domain-selected, labels are single-annotator, category support is uneven, deterministic natural-language mapping is limited, benchmark results do not consistently exceed simple baselines, and the historical and generated benchmark snapshots differ.

## Known Test Limitations

The focused Phase 7C suite passes 20 tests. The complete DevLens suite has 30 passed and one pre-existing unrelated fork-enrichment failure. The complete GDERS suite was attempted but did not produce a verifiable completion summary.

## Recommended Future Research Directions

Future work may expand repository coverage, add independent annotators, calibrate classifier confidence, evaluate semantic query mapping, reconcile benchmark artifact versioning, and improve test-environment reproducibility. These are recommendations only; none are implemented in Phase 7D.

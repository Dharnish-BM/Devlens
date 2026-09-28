# Phase 7D GDERS Data Flow

| Stage | Input | Output | Persisted artifact | Training use | Evaluation use | User-facing |
|---|---|---|---|---|---|---|
| Repository selection | Frozen target repository list | 10 selected repositories | `gders/config.py`, `data/gders/processed/dataset_manifest.json` | Defines corpus | Defines evaluation domain | No |
| PR collection | Repository names and GitHub API responses | 333 collected pull requests | `data/gders/raw_comments/`, collection manifest | Source data | Source data | No |
| Review-comment extraction | Collected pull requests | 1,225 review comments | Raw comment JSONL files | Source data | Source data | No |
| NLP preprocessing | Raw comment bodies | Normalized text and metadata | `data/gders/processed/comments/`, `preprocessing_report.json` | Classifier features | Inputs to frozen inference | No |
| Annotation | Preprocessed comment sample | 341 gold labels, including 185 Phase 5A and 156 Phase 5C labels | `expanded_gold_dataset.jsonl`, annotation reports | Classifier training/reference | Held-out relevance labels | No |
| Classifier training | Labeled comments and taxonomy | Frozen LinearSVC decision-margin classifier | `data/gders/models/`, model comparison reports | Yes | Frozen model used for inference | No |
| Full-corpus inference | Preprocessed comments and frozen classifier | 884 inferred comment records with confidence tiers | `comment_predictions.jsonl` | No | Supplies permitted profile evidence | No |
| Evidence scoring | Gold and permitted inferred records | Developer/category evidence scores and tiers | `developer_expertise_profiles.jsonl`, `expertise_profile_report.json` | No | Candidate construction | Indirectly |
| Identity classification | Reviewer logins and corpus signals | Human, bot/service, or uncertain identity classes | `reviewer_identity_audit.json` | No | Candidate identity gate | No |
| Recommendation eligibility | Evidence tiers plus identity class | Eligible developer/category candidates | Profile artifact and recommender report | No | Candidate filtering | Yes through recommendation view |
| Recommendation | Expertise category or deterministic natural-language query | Ranked human candidate list with evidence traceability | Recommendation response; report artifact | No | Phase 7B retrieval evaluation | Yes through GDERS view |
| Held-out evaluation | Developer-grouped gold evidence and frozen profiles | 142 query results, metrics, baselines, ablations | `benchmark_dataset.jsonl`, `benchmark_results.json`, `benchmark_leakage_audit.json` | No | Yes | No |
| Dashboard integration | DevLens username and frozen GDERS artifacts | Read-only GDERS evidence/recommendation presentation | `gders/integration/devlens_bridge.py`, dashboard template | No | No new benchmark evaluation | Yes |

## Leakage Controls

For each benchmark query, the highest eligible gold comment ID is held out by developer/category grouping. Held-out IDs are excluded from profile construction and checked against profile supporting evidence. Relevance labels come from held-out gold labels, never from recommendation output. The recorded leakage audit reports zero violations across 142 queries.

## Boundary Controls

The dashboard bridge passes only GitHub username from DevLens to GDERS. GDERS returns presentation data only. It does not mutate DevLens storage or model inputs, and it does not alter `/match`.

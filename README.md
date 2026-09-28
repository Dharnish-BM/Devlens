# devlens

ML-based GitHub developer skill profiler

## DevLens and GDERS Research System

### What is DevLens?

DevLens is the GitHub developer intelligence system. It collects profile and repository signals, derives standardized features, applies the existing K-Means and XGBoost models, produces SHAP explanations, and supports the existing project-fit `/match` workflow.

### What is GDERS?

GDERS is the GitHub Developer Expertise Recommendation System. It analyzes pull-request review comments using a ten-category code-review taxonomy, NLP preprocessing, a multilabel Linear SVM classifier, evidence profiles, identity filtering, and a deterministic recommender.

### How are they related?

Phase 7C connects the systems at the dashboard application layer. DevLens passes a GitHub username to `DevLensGDERSBridge`; the bridge reads frozen GDERS artifacts and returns separate evidence and recommendation presentation data. GDERS does not feed back into DevLens features, ML training, database tables, SHAP, project-fit scoring, or `/match`.

### What evidence does GDERS use?

GDERS uses labeled and permitted confidence-tiered PR review-comment evidence. Gold comments, high-confidence predictions, and medium-confidence predictions can contribute to evidence profiles under the frozen contract. Low-confidence and abstained predictions are retained for audit but do not establish recommendation eligibility.

### How is developer expertise determined?

Expertise is a bounded association with observed review evidence in one or more taxonomy categories. It is not a complete assessment of a person's capabilities. Category scores combine evidence weights with PR and repository diversity factors, and recommendations require sufficient evidence plus the identity gate.

### How are bots handled?

Bot and service-account evidence remains available for corpus auditing, but accounts classified as `bot_or_service_account` or `uncertain` are excluded from human developer recommendations. Current audited examples include `Copilot` and `gemini-code-assist[bot]`.

### What happens when evidence is insufficient?

The dashboard returns `INSUFFICIENT_PR_EXPERTISE_EVIDENCE` and does not infer expertise from DevLens archetypes, programming languages, repository languages, GitHub activity, generic profile data, or project-fit scores.

### How are recommendations evaluated?

Phase 7B uses developer-grouped deterministic held-out gold evidence. Held-out comments are excluded from profile construction, relevance labels are derived independently from held-out labels, and a leakage audit checks every query. Results are reported with P@1/P@3/P@5, R@1/R@3/R@5, MRR, nDCG@5, simple baselines, and evidence ablations.

### What are the limitations?

The corpus contains 10 selected repositories and 1,225 review comments. Category support is uneven, only six multi-category benchmark queries are available, labels are single-annotator, natural-language mapping is deterministic, classifier margins are not calibrated probabilities, and the benchmark does not consistently outperform simple baselines. The complete GDERS regression suite has no verified completion result, and one unrelated DevLens fork-enrichment test remains failing.

### How do I reproduce the experiments?

Use the frozen artifacts and the environment recorded in [phase7d_reproducibility_manifest.json](phase7d_reproducibility_manifest.json). The research packaging is documented in [PHASE7D_ARCHITECTURE.md](PHASE7D_ARCHITECTURE.md), [PHASE7D_DATA_FLOW.md](PHASE7D_DATA_FLOW.md), [PHASE7D_RESULTS.md](PHASE7D_RESULTS.md), [PHASE7D_REPRODUCIBILITY_CHECKLIST.md](PHASE7D_REPRODUCIBILITY_CHECKLIST.md), and [PHASE7D_ARTIFACT_INDEX.md](PHASE7D_ARTIFACT_INDEX.md). Do not recollect data or regenerate benchmark artifacts unless starting a separately identified experiment.

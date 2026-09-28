# Phase 7D Architecture

## System Boundary

DevLens and GDERS are separate systems. Phase 7C adds an application-layer, read-only bridge. GDERS expertise evidence is not written into DevLens feature vectors, ML tables, training data, or project-fit scoring.

## DevLens

```text
GitHub
  -> collection
  -> 52 raw metrics
  -> 41 standardized features
  -> K-Means clustering
  -> XGBoost classification
  -> SHAP explanations
  -> Developer Intelligence
```

The existing DevLens pipeline persists profile snapshots and model outputs through its existing database and artifact paths. The existing `/match` route and project-fit scoring remain separate from GDERS recommendations.

## GDERS

```text
GitHub repositories
  -> pull requests
  -> PR review comments
  -> NLP preprocessing
  -> ten-category taxonomy
  -> multilabel classification
  -> evidence profiles
  -> identity gate
  -> expertise recommender
```

The taxonomy categories are `ARCH_DESIGN`, `BUG_LOGIC`, `CODE_STYLE`, `DATA_MANAGEMENT`, `DOCUMENTATION`, `FRONTEND_UI_UX`, `INFRA_DEVOPS`, `PERF_OPTIMIZATION`, `SECURITY_PRIVACY`, and `TESTING_QUALITY`.

The classifier is a One-vs-Rest Linear SVM represented by decision margins. Margin tiers are not calibrated probabilities. Gold evidence, high-confidence predictions, and medium-confidence predictions can contribute to recommendation eligibility under the frozen contract; low-confidence and abstained predictions are audit-only.

## Integration

```text
DevLens GitHub username
  -> DevLensGDERSBridge
  -> frozen GDERS profile and evidence artifacts
  -> read-only normalized dashboard data
  -> integrated developer dashboard
```

The bridge accepts GitHub username as the cross-system identity. It can display evidence metadata and invoke the existing GDERS recommender for a separate recommendation view. Missing or non-qualifying evidence is represented explicitly as `INSUFFICIENT_PR_EXPERTISE_EVIDENCE`.

Identity filtering requires `identity_class == human_candidate` and `developer_recommendation_eligible == true`. Bot and service-account evidence remains retained for audit but is excluded from human recommendations.

## Non-Feedback Boundary

GDERS does not feed back into DevLens ML. No GDERS category, evidence, confidence score, reviewer profile, or recommendation is added to DevLens's 41 features, training data, database schema, K-Means inputs, XGBoost inputs, SHAP computation, or project-fit score.

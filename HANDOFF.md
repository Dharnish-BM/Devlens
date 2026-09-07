# DevLens — Project Handoff & Definitive Technical Reference

> **Definitive Master Reference Document (Phases 0–10 Complete)**  
> **Repository:** `d:\GIT\Devlens` | **Active Cohort:** 188 Consented GitHub Developers  
> **Database:** SQLite (`devlens.db`) | **Status:** All Pipelines, Models, Analytics, and Web UI Deployed and Verified.

---

## 1. Project Status & Cohort Integrity

DevLens is an end-to-end developer intelligence, empirical profiling, functional archetype classification, and project-fit matchmaking system evaluated on an empirical cohort of undergraduate computer science engineers.

### Cohort Accounting (Final $N=188$)
*   **Total Consented Classmate Profiles Processed:** **188 valid developers**
*   **Excluded Profiles (4 total):**
    1.  `anbuchelvan-efx`: Unresolvable/invalid GitHub username (HTTP 404).
    2.  `"Suhitha M"`: Unresolvable handle string (no valid GitHub account resolved).
    3.  `"Abisha_Rebekkal"`: Malformed username string with underscore (unresolvable 404).
    4.  `torvalds`: Synthetic pipeline sanity test profile (Linus Torvalds, excluded from student cohort statistics).
*   **CRITICAL IDENTITY DISTINCTION:**
    *   `Abisha-Rebekkal21` is a **valid, active student profile** with 12 repositories and is **fully included** in the 188-developer dataset.
    *   Do **NOT** confuse `Abisha-Rebekkal21` with the excluded malformed entry `"Abisha_Rebekkal"`.

---

## 2. Raw Data Schema (`data/raw/{username}.json`)

All profiles were collected using GitHub REST API v3 and GraphQL API v4. Each profile JSON contains the following complete top-level structure:

```json
{
  "username": "string",
  "collected_at": "ISO-8601 UTC timestamp (e.g. 2026-08-13T03:48:26Z)",
  "user_info": {
    "login": "string",
    "name": "string or null",
    "bio": "string or null",
    "blog": "string or null",
    "public_repos": "integer",
    "followers": "integer",
    "following": "integer",
    "created_at": "ISO-8601 timestamp",
    "updated_at": "ISO-8601 timestamp"
  },
  "repositories": [
    {
      "name": "string",
      "full_name": "string",
      "is_fork": "boolean",
      "forks_count": "integer",
      "stargazers_count": "integer",
      "size": "integer (KB)",
      "language": "string or null",
      "created_at": "ISO-8601 timestamp",
      "updated_at": "ISO-8601 timestamp",
      "pushed_at": "ISO-8601 timestamp",
      "description": "string or null",
      "readme_length_chars": "integer (Phase 1B enriched, 0 for forks)",
      "has_ci_config": "boolean (.github/workflows, .travis.yml, etc., False for forks)",
      "has_test_presence": "boolean (test_*, *_test.*, .spec.*, False for forks)",
      "recent_commit_messages": ["list of strings (up to 30 commit subjects, [] for forks)"]
    }
  ],
  "commit_activity": {
    "total_commits_sampled": "integer",
    "weekly_commit_counts": ["52 integers representing commits per week"]
  },
  "pull_requests": {
    "total_opened": "integer",
    "total_merged": "integer",
    "review_participation_count": "integer"
  },
  "issues": {
    "total_opened": "integer",
    "total_closed": "integer",
    "total_commented": "integer"
  },
  "graphql_data": {
    "pinned_repositories": ["list of up to 6 pinned repo names"],
    "contribution_calendar": {
      "total_contributions": "integer",
      "active_days_count": "integer",
      "longest_streak_days": "integer"
    },
    "languages_breakdown": {
      "repo_name": [
        {
          "language": "string (e.g. JavaScript, C++, Dockerfile)",
          "size_bytes": "integer"
        }
      ]
    }
  }
}
```

---

## 3. Feature Engineering — Final Locked Formulas

All feature engineering logic is defined in `devlens/features/feature_engineering.py`.

### 3a. Documentation & Engineering Maturity Scoring Weights

Both weight dictionaries are runtime-asserted to sum to $1.00$ at module import.

#### `DOC_WEIGHTS` (`feature_engineering.py:129-135`)
```python
DOC_WEIGHTS = {
    "desc_coverage": 0.30,  # Fraction of original repos with non-empty description string
    "desc_depth":    0.25,  # Mean normalized README length: mean(min(readme_chars, 5000) / 5000)
    "bio":           0.15,  # 1.0 if user has non-empty bio string, else 0.0
    "blog":          0.10,  # 1.0 if user has non-empty blog/website URL, else 0.0
    "pinned":        0.20,  # Pinned repo showcase ratio: min(len(pinned_repos), 6) / 6.0
}
```

#### `ENG_WEIGHTS` (`feature_engineering.py:132-138`)
```python
ENG_WEIGHTS = {
    "ci":                   0.30,  # Fraction of original repos with CI config files (has_ci_config)
    "test":                 0.25,  # Fraction of original repos with test presence (has_test_presence)
    "commit_message":       0.20,  # Fraction of sampled commit subjects matching Conventional Commits regex
    "pr_disc":              0.15,  # PR merge discipline ratio: total_merged / max(total_opened, 1)
    "review_participation": 0.10,  # 1.0 if review_participation_count > 0, else 0.0
}
```

### 3b. Complete Canonical Field-Name Registry (Locked — Do Not Rename)

| Canonical Name | Domain | Formula / Measure | Range |
|---|---|---|---|
| `doc_desc_coverage` | Documentation | Fraction of original repos with non-null `description` | $[0.0, 1.0]$ |
| `doc_desc_depth_mean` | Documentation | Mean normalized README length: $\text{mean}(\min(L, 5000)/5000)$ | $[0.0, 1.0]$ |
| `doc_has_bio` | Documentation | Binary flag: $1.0$ if profile `bio` is non-empty | $\{0.0, 1.0\}$ |
| `doc_has_blog` | Documentation | Binary flag: $1.0$ if profile `blog` is non-empty | $\{0.0, 1.0\}$ |
| `doc_pinned_count` | Documentation | Integer count of pinned repositories | $[0, 6]$ |
| `doc_pinned_ratio` | Documentation | Pinned showcase ratio: $\text{count} / 6.0$ | $[0.0, 1.0]$ |
| `doc_score` | Documentation | Weighted linear composite of above 5 metrics | $[0.0, 1.0]$ |
| `eng_ci_ratio` | Engineering | Fraction of original repos with CI workflows | $[0.0, 1.0]$ |
| `eng_test_ratio` | Engineering | Fraction of original repos with test suites | $[0.0, 1.0]$ |
| `eng_commit_message_quality` | Engineering | Fraction of commit subjects matching structured regex | $[0.0, 1.0]$ |
| `eng_pr_discipline` | Engineering | PR merge rate: `total_merged / max(total_opened, 1)` | $[0.0, 1.0]$ |
| `eng_review_participation` | Engineering | Binary flag: $1.0$ if `review_participation_count > 0` | $\{0.0, 1.0\}$ |
| `eng_maturity_score` | Engineering | Weighted linear composite of above 5 metrics | $[0.0, 1.0]$ |
| `repo_has_stars` | Repository | Binarized flag: $1.0$ if `repo_total_stars > 0` | $\{0.0, 1.0\}$ |
| `collab_issues_opened` | Collaboration | Binarized flag: $1.0$ if `issues_opened > 0` | $\{0.0, 1.0\}$ |
| `collab_issues_commented` | Collaboration | Binarized flag: $1.0$ if `issues_commented > 0` | $\{0.0, 1.0\}$ |
| `lang_mobile_signal` | Technology | $\text{mobile\_repos} / N_{\text{original}}$ (`{"Swift", "Dart", "Kotlin", "Objective-C"}`) | $[0.0, 1.0]$ |

### 3c. Crucial Bug Fixes Applied (Root Causes Documented — Do Not Regress)
1.  **Fork Exclusion Guard (`collect_profile.py` & `feature_engineering.py`):**
    *   *Bug:* Forked repositories distorted `repo_mean_size_kb`, `repo_years_active`, `eng_ci_ratio`, and `eng_test_ratio` with upstream codebases.
    *   *Fix:* All repo-level feature loops strictly filter to `original = [r for r in repos if not r.get("is_fork")]`.
2.  **GitHub 409 Conflict on Empty Repositories:**
    *   *Bug:* Repositories initialized without commits returned HTTP 409 when querying commit subjects.
    *   *Fix:* Explicit try/except handling assigns `recent_commit_messages = []` on 409 responses.
3.  **Documentation Coverage Contamination:**
    *   *Bug:* `doc_desc_coverage` previously checked language names or README fallbacks.
    *   *Fix:* Strictly inspects `repo.get("description")` only.
4.  **Language Proxy Removal in CI/Test Metrics:**
    *   *Bug:* `eng_ci_ratio` and `eng_test_ratio` used language proxies when keys were missing.
    *   *Fix:* Uses explicit Phase 1B AST/tree inspection keys (`has_ci_config` and `has_test_presence`).
5.  **Sparse Feature Binarization:**
    *   *Problem:* Continuous count features (`repo_total_stars`, `collab_issues_opened`, `collab_issues_commented`) have $>90\%$ zero-values across the cohort. Standard scaling gave rare non-zero users extreme $Z$-scores ($Z > 5.0$), distorting distance metrics and tree splits.
    *   *Fix:* Binarized to $(>0 \rightarrow 1.0)$.
6.  **Zero-Variance Column Drop:**
    *   `lang_primary_is_go` and `lang_primary_is_rust` are $0.0$ for all 188 students and are dropped before scaling to prevent divide-by-zero errors.

---

## 4. The 41-Feature Scaled Clustering & Modeling Matrix

Prepared via `prepare_scaled_feature_matrix(drop_zero_variance=True)`.

### Included Features (41 Total):
1.  `activity_active_weeks_ratio`
2.  `activity_calendar_days_active`
3.  `activity_longest_streak_days`
4.  `activity_max_week`
5.  `activity_total_commits`
6.  `activity_weekly_cv`
7.  `activity_weekly_mean`
8.  `activity_weekly_std`
9.  `collab_issues_commented` *(Binarized)*
10. `collab_issues_opened` *(Binarized)*
11. `collab_pr_merged`
12. `collab_pr_opened`
13. `doc_desc_coverage`
14. `doc_desc_depth_mean`
15. `doc_has_bio`
16. `doc_has_blog`
17. `doc_pinned_count`
18. `doc_pinned_ratio`
19. `eng_ci_ratio`
20. `eng_commit_message_quality`
21. `eng_pr_discipline`
22. `eng_review_participation`
23. `eng_test_ratio`
24. `lang_devops_signal`
25. `lang_diversity_entropy`
26. `lang_frontend_signal`
27. `lang_ml_signal`
28. `lang_mobile_signal`
29. `lang_primary_is_c`
30. `lang_primary_is_java_kotlin`
31. `lang_primary_is_javascript`
32. `lang_primary_is_python`
33. `lang_unique_count`
34. `repo_fork_ratio`
35. `repo_mean_size_kb`
36. `repo_original_count`
37. `repo_std_size_kb`
38. `repo_total_count`
39. `repo_total_forks_received`
40. `repo_years_active`
41. `repo_has_stars` *(Binarized)*

### Excluded Features & Rationale (10 Total):
*   `doc_score` & `eng_maturity_score`: Excluded because their constituent sub-components are already present in the matrix (avoids double-counting composite variance).
*   `collab_pr_merge_rate`: Excluded as a direct mathematical quotient of `collab_pr_opened` and `collab_pr_merged`.
*   `collab_issue_close_rate` & `collab_issues_closed`: Excluded to prevent triple-counting the issues opened signal.
*   `collab_review_count` & `collab_review_rate_per_pr`: Excluded; replaced by `eng_review_participation`.
*   `repo_max_stars`, `repo_mean_stars`, `repo_total_stars`: Excluded; replaced by `repo_has_stars`.

---

## 5. Phase 5 K-Means Clustering — Findings & Causal Chain

Evaluated across $k=2..10$ using Silhouette Scores and Inertia (Elbow Method):

```
k     Inertia         Silhouette Score    Status
-------------------------------------------------------
2     6761.0028       0.3865              Optimal / Selected
3     6288.7992       0.1226              
4     5975.3710       0.0810              
5     5746.9513       0.0677              
6     5468.4880       0.0648              
7     5286.9329       0.2046              
8     5027.0468       0.0673              
9     4947.3440       0.0655              
10    4670.5334       0.0459              
```

### Empirical Cluster Structure ($k=2$):
*   **Cluster 0: Power-User / Showcase Developers (17 developers, 9.0%):**
    Distinguished by extreme activity cadence: $+2.28\,\sigma$ calendar active days (136.8 vs 25.4 mean), $+2.27\,\sigma$ total commits (616.2 vs 89.0 mean), and $+1.85\,\sigma$ pinned repo depth.
*   **Cluster 1: Baseline Developers (171 developers, 91.0%):** Standard student course activity profile.

### Causal Chain to Phase 6:
1.  Diagnostic clustering runs on non-activity features (32 features) and language-only features (9 features) confirmed that **no natural, balanced functional-role clusters exist in this cohort**.
2.  Under a strict $\ge 5\%$ (min 10 developers) cluster validity threshold, any $k \ge 3$ creates invalid singleton or micro-clusters (1–8 developers).
3.  **Root Cause:** Extreme technology concentration (**70.7% of students are JavaScript-primary**).
4.  **Methodological Decision:** Unsupervised K-Means measures **Engagement Tier** (Activity/Presentation Intensity), whereas **Functional Archetypes** must be classified via explicit domain rules (Phase 6).

---

## 6. Phase 6 — Archetype Classification & The Label Leakage Finding

### 6a. Deterministic Precedence-Ordered Archetype Rules

Evaluated in strict hierarchical order to resolve multi-stack overlaps:

```
1. ML Specialist:
   is_valid_specialized_signal(lang_ml_signal, base=0.05, comp=top_competing_ml, ceil=0.100, D=0.55)

2. Mobile Developer:
   is_valid_specialized_signal(lang_mobile_signal, base=0.02, comp=top_competing_mob, ceil=0.100, D=0.55)

3. DevOps Engineer:
   is_valid_specialized_signal(lang_devops_signal, base=0.10, comp=top_competing_devops, ceil=0.120, D=0.55)

4. Frontend Developer:
   lang_primary_is_javascript == 1.0 OR lang_frontend_signal > 0.50

5. Backend Developer:
   lang_primary_is_c == 1.0 OR (lang_primary_is_java_kotlin == 1.0 AND lang_mobile_signal <= 0.02)

6. Full-Stack Developer:
   lang_diversity_entropy >= 1.8 (Genuine polyglot multi-domain breadth)

7. Unclassified / Low Signal:
   Fallback for low entropy (< 1.8) and unclassifiable / inactive profiles
```

#### Competing Primary Language Dominance Guard & Dynamic Ceiling:
To prevent incidental cross-stack traces in large or polyglot portfolios from dominating an engineer's primary career work (such as a single Swift macOS utility labeling a 64% Python engineer as a "Mobile Developer"), specialized tiers (ML, Mobile, DevOps) are guarded by `is_valid_specialized_signal()`:

$$\text{ceiling} = \max\Big(\text{min\_ceiling}=0.10, \ \text{base\_threshold} \times 1.20\Big)$$

*   **Dedicated Specialization ($\text{signal} \ge \text{ceiling}$):** Immune to competing stack suppression (e.g. Mobile $\ge 0.100$, ML $\ge 0.100$, DevOps $\ge 0.120$).
*   **Weak Specialized Signal ($\text{signal} < \text{ceiling}$):** Suppressed if any single competing primary language outside the category holds $> 55.0\%$ of original repositories ($\text{Dominance Threshold } D = 0.55$, empirically set strictly above the cohort's 90th percentile of $54.85\%$).
*   **Resolved False-Positive Cases:**
    1.  **`kennethreitz` (Live Ingestion, $N_{\text{orig}}=47$):** 30 Python repos ($63.83\%$ share $> 55\%$) and 1 incidental Swift macOS desktop client (`RetroVault`, $\text{mob\_sig} = 0.0213 < 0.100$). The dominance guard suppresses the false Mobile classification and assigns him to **`DevOps Engineer`** ($\text{devops\_sig} = 0.4894 \ge 0.120$, reflecting his 23 Docker/Makefile toolchain repos across `pipenv`, `bake`, `responder`).
    2.  **Synthetic 35-Repo Case ($N_{\text{orig}}=35$):** 22 Python repos ($62.86\% > 55\%$) and 1 Swift repo ($\text{mob\_sig} = 0.0286 < 0.100$). The guard correctly suppresses Mobile, proving invariance across all portfolio sizes.
    3.  **`Praneshbabu982005-hub` (Cohort Boundary Preservation, $N_{\text{orig}}=7$):** 4 JavaScript repos ($57.14\%$) and 1 Shell repo ($\text{devops\_sig} = 0.1429 \ge 0.120$). The dynamic ceiling correctly recognizes this as dedicated tooling work, ensuring **$0$ drift across all 188 cohort developers**.

### Final Cohort Distribution ($N=188$):
*   **Frontend Developer:** 109 (58.0%) — *Trained in XGBoost*
*   **DevOps Engineer:** 41 (21.8%) — *Trained in XGBoost*
*   **Mobile Developer:** 13 (6.9%) — *Trained in XGBoost*
*   **ML Specialist:** 13 (6.9%) — *Trained in XGBoost*
*   **Unclassified / Low Signal:** 9 (4.8%) — *Trained in XGBoost*
*   **Backend Developer:** 2 (1.1%) — *Excluded from training, preserved as DB Ground Truth*
*   **Full-Stack Developer:** 1 (0.5%) — *Excluded from training, preserved as DB Ground Truth*

---

### 6b. CRITICAL METHODOLOGICAL FINDING: Label Leakage vs. Behavioral Prediction

#### The Leakage Phenomenon:
When XGBoost is trained on all 41 features (including the 8 language features used to define the pseudo-labels), it achieves **`96.43%` held-out test accuracy** (70/30 split) and **`96.76% ± 3.15%` 5-fold cross-validation accuracy**.

However, SHAP feature attribution and ablation experiments prove this is **rule-reproduction**, not behavioral discovery:
1.  **SHAP Attribution:** **`86.44%` of total SHAP magnitude** comes exclusively from the 8 labeling-source features (`lang_ml_signal`, `lang_mobile_signal`, `lang_devops_signal`, `lang_frontend_signal`, etc.).
2.  **Ablation Study (Removing the 8 Labeling Features):**
    *   When trained purely on the remaining 33 behavioral/quality features (activity, PR discipline, commit quality, description depth, stars), **test accuracy collapses to `48.21%`** (below the 58.9% majority-class baseline).
    *   ML Specialist recall drops to **`0.0%`** across all 5 folds.
3.  **Academic Conclusion for the Paper:**
    Functional engineering specialization in early-career developers is **fundamentally defined by explicit language and framework choice**, and is **not recoverable from activity cadences or repository hygiene alone**. Report the 96.4% figure honestly as rule internalization and explainability verification, not as independent behavioral prediction.

---

## 7. Phase 8 — Project-Fit Matchmaking & Benchmark Findings

### 7a. Continuous Matchmaking Formula

$$\text{FitScore}(d, R) = w_{\text{arch}} \cdot S_{\text{arch}} + w_{\text{lang}} \cdot S_{\text{lang}} + w_{\text{mat}} \cdot S_{\text{mat}}$$

1.  **Archetype Component ($S_{\text{arch}}$):** Maximum credit from the technical-adjacency table:
    *   Exact Domain Match: **`1.00`**
    *   Full-Stack matching any specialist requirement: **`0.60`**
    *   Frontend / Backend matching Full-Stack: **`0.50`**
    *   DevOps / Mobile / ML matching Full-Stack: **`0.40`**
    *   DevOps $\leftrightarrow$ Backend Adjacency: **`0.40`**
    *   Frontend $\leftrightarrow$ Mobile UI Adjacency: **`0.30`**
    *   Backend $\leftrightarrow$ ML Python/Data Adjacency: **`0.30`**
    *   Unclassified / Low Signal $\rightarrow$ Any: **`0.00`**
2.  **Language Component ($S_{\text{lang}}$):** Cosine similarity between developer proportional byte-distribution vector $\mathbf{u}$ (from raw JSON `languages_breakdown`) and requirement vector $\mathbf{v}$. Guaranteed $0.0$ fallback for zero-byte accounts.
3.  **Quality / Maturity Component ($S_{\text{mat}}$):** Continuous normalized distance score:
    $$S_{\text{mat}} = \frac{\min\left(1.0, \frac{\text{DocScore}}{D_{\min}}\right) + \min\left(1.0, \frac{\text{EngScore}}{E_{\min}}\right)}{2}$$

### 7b. Benchmark Comparison Findings (vs. Naive GDERS Keyword Baseline):
1.  **Resolution of Binary Ties (Frontend Benchmark):**
    *   Naive Keyword Baseline returned **96 undifferentiated binary matches** ($S=1$).
    *   DevLens produced a granular continuous ranking (#1 `dineshkumarp07`: `0.8741`, #2 `Tharuniga60`: `0.8516`, #3 `Arun2005s`: `0.8348`) separating active, well-documented developers from empty accounts.
2.  **Mitigation of Keyword False Negatives (DevOps Benchmark):**
    *   Naive Keyword Baseline **collapsed to only 2 matches**, missing 39 valid DevOps engineers because their bulk codebase byte volume was in JavaScript/HTML.
    *   DevLens identified top candidates (`vijeth06` #1: `0.6439`, `dharshini-akb` #4: `0.5993`) based on real Dockerfile/CI configurations and engineering maturity.
3.  **Substantial Backend Exemplar (`RBROHANTH`):**
    *   Verified as a high-volume systems developer: **40 original repositories, 697 MB total code, 85.6% C++, 13.6% C**, SIH hackathon projects. Ranks #1 in Backend requirements (`FitScore = 0.7455`), confirming the match is not a sparse-profile artifact.

---

## 8. Phase 9 — Growth Trajectory & Snapshot Integrity

### 8a. Longitudinal Trend Analysis (`devlens/models/growth_trajectory.py`)
*   `get_growth_summary(username)` calculates linear regression slopes ($\Delta \text{score}/\text{month}$), feature deltas, and archetype stability when $\ge 2$ collection snapshots exist.
*   When only 1 collection snapshot exists, it cleanly returns:
    ```json
    {
      "status": "insufficient_history",
      "message": "Only one collection snapshot available; trend analysis requires repeat collections over time.",
      "snapshot_count": 1
    }
    ```

### 8b. Database Snapshot Bug & Cleanup Fix (CRITICAL KNOWLEDGE)
*   **The Trap / Bug:** In earlier pipeline phases, running batch feature engineering repeatedly created new `Snapshot` rows in `devlens.db` without fetching new GitHub data, causing 938 redundant snapshot rows to accumulate.
*   **The Permanent Fix:**
    1.  The database was cleaned up, deleting all intermediate rows.
    2.  `devlens.db` now contains **exactly 188 developers and exactly 188 snapshots (1:1 parity)**.
    3.  `get_growth_summary()` verified on real users (`dineshkumarp07`, `vijeth06`, `Jeevashre12`, `AMARNATH002`, `RBROHANTH`), correctly returning `insufficient_history`.
    4.  **Rule for Future Sessions:** *Entries in the `snapshots` table must represent genuinely new GitHub REST/GraphQL fetch events with new `collected_at` dates from raw JSON. Re-running feature engineering or model training must update existing snapshot child rows, never insert new snapshot rows.*

---

## 9. Phase 10 — Flask Application & UI Design System

*   **Location:** `devlens/app/app.py` (Templates in `devlens/app/templates/`, CSS in `devlens/app/static/css/`)
*   **Design Tokens:**
    *   **Typography:** `Space Grotesk` (Headings), `IBM Plex Sans` (Body text), `IBM Plex Mono` (Data, scores, diffs, code).
    *   **Palette:** Ink `#0d1117`, Paper `#161b22`, Signal Green `#3fb950`, Diff Red `#f85149`, Amber `#e3b341`, Slate `#8b949e`.
    *   **Visual Styling:** Git commit-log layout, monospace snapshot metadata, diff-style `+`/`-` SHAP bars.
*   **Served Routes:**
    1.  `GET /`: Searchable directory across all 188 developers with archetype and cluster filters. Displays academic showcase disclaimer.
    2.  `GET /profile/<username>`: Developer inspection page with Engagement Tier, Archetype, Doc/Eng maturity breakdown, SHAP diff bars, language byte charts, and growth notice.
    3.  `GET /match`: Interactive matchmaking tool with side-by-side DevLens vs. GDERS baseline comparison (presets for DevOps, Frontend, Backend).
    4.  `GET /upload` & `POST /upload/process`: Resume ingestion (single PDF/DOCX or shared Drive folder) with multi-strategy GitHub username discovery and staging.
    5.  `POST /collect/start/<username>`, `GET /collect/status/<job_id>`, `GET /collect/progress/<job_id>`: Asynchronous collection pipeline executing Phases 1–7 in the background with real-time stage streaming and tagging new candidates with `source="live_upload"`.

---

## 10. Known Limitations (For Academic Paper)

1.  **Single Collection Wave:** Growth trajectory logic is structurally implemented and tested, but unvalidated against multi-year real-world longitudinal data.
2.  **Rule-Derived Archetypes:** Archetypes are grounded in explicit technology thresholds rather than unsupervised discovery due to dataset JavaScript dominance.
3.  **Ultra-Minority Classes:** `Backend Developer` ($n=2$) and `Full-Stack Developer` ($n=1$) were preserved in the database as heuristic ground truth but excluded from XGBoost training.
4.  **Baseline Comparator Scope:** The rule-based benchmark comparator simulates GDERS's category-matching philosophy (exact keyword filtering) rather than a full re-implementation of GDERS's BERT-based text embedding stack.

---

## 11. Pipeline Phase Summary Registry

| Phase | Description | Final State |
|---|---|---|
| **Phase 0** | Project Scaffolding & Directory Setup | **COMPLETE** — Standard modular package structure initialized. |
| **Phase 1** | GitHub REST API Profile Collection | **COMPLETE** — Rate-limited collection with fork filtering & 409 handling. |
| **Phase 1B**| Deep Profile Enrichment (AST/CI/Tests) | **COMPLETE** — README length, CI workflow, test file, and commit message collection. |
| **Phase 2** | Resume Parser & Skill Extraction | **COMPLETE** — PDF/DOCX resume text extraction and skill keyword matching. |
| **Phase 3** | SQLite Database & Repository Layer | **COMPLETE** — SQLAlchemy models (`devlens.db`) with clean CRUD repository. |
| **Phase 4** | Feature Engineering & Scaling Pipeline | **COMPLETE** — 52 features computed, 41 scaled features, binarization applied. |
| **Phase 5** | K-Means Empirical Clustering | **COMPLETE** — $k=2$ selected (Silhouette 0.39), separating Power-User vs. Baseline. |
| **Phase 6** | XGBoost Archetype Classification & SHAP | **COMPLETE** — 5-class model trained ($96.4\%$ test acc), leakage verified ($86.4\%$ SHAP). |
| **Phase 7** | Longitudinal Growth Trajectory | **COMPLETE** — Linear slope calculation & single-snapshot guard implemented. |
| **Phase 8** | Project-Fit Matchmaking Engine | **COMPLETE** — Multi-component continuous scoring benchmarked against GDERS. |
| **Phase 9** | Snapshot Database Normalization | **COMPLETE** — Purged 938 redundant snapshot rows, restored exact 1:1 parity. |
| **Phase 10**| Web UI, Resume Ingestion & Async Pipeline | **COMPLETE** — Commit-log UI, multi-tier resume parsing, and async collection worker. |

---
*End of Master Reference Document.*

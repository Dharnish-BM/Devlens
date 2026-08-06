# DevLens Data Pipeline — Handoff Reference

**Generated:** 2026-08-06  
**Purpose:** Authoritative state document for continuing work in a different tool or session.  
Code is in `d:\GIT\Devlens`. Python 3.11.4 venv at `venv/`. SQLite DB at `devlens.db`.

---

## 1. Raw Data Schema

**File location:** `data/raw/{username}.json`  
**Produced by:** `devlens/data_collection/collect_profile.py` → `collect_full_profile()` + `enrich_repo_details()`

```json
{
  "username": "string",
  "collected_at": "2026-08-06T08:41:47.927317Z",

  "user_info": {
    "login": "string",
    "name": "string",
    "bio": "string | null",
    "blog": "string | null",
    "public_repos": 12,
    "followers": 0,
    "following": 0,
    "created_at": "2011-09-05T..."
  },

  "repositories": [
    {
      "id": 2325298,
      "name": "linux",
      "full_name": "torvalds/linux",
      "description": "Linux kernel source tree",
      "language": "C",
      "stars": 241915,
      "forks": 64738,
      "size": 5205380,
      "created_at": "2011-09-04T22:48:20Z",
      "pushed_at": "2026-08-05T21:00:00Z",
      "is_fork": false,
      "default_branch": "master",

      "readme_length_chars": 6043,
      "has_ci_config": false,
      "has_test_presence": false,
      "recent_commit_messages": [
        "Merge tag 'soc-fixes-7.2-2' of ...",
        "mm: fix incorrect flush address..."
      ]
    }
  ],

  "commit_activity": {
    "linux": [
      {"days": [0,0,0,0,0,0,5], "total": 5, "week": 1720742400}
    ]
  },

  "pull_requests": {
    "total_opened": 85,
    "total_merged": 72,
    "review_participation_count": 34
  },

  "issues": {
    "total_opened": 10,
    "total_closed": 9,
    "total_commented": 43
  },

  "graphql_data": {
    "contribution_calendar": {
      "totalContributions": 3275,
      "weeks": [
        {
          "contributionDays": [
            {"date": "2026-01-05", "contributionCount": 12, "color": "#216e39"}
          ]
        }
      ]
    },
    "pinned_repositories": [
      {
        "name": "linux",
        "description": "Linux kernel source tree",
        "stars": 241915,
        "forks": 64738,
        "primary_language": "C"
      }
    ],
    "languages_breakdown": {
      "linux": [
        {"language": "C", "size_bytes": 1048576000},
        {"language": "Shell", "size_bytes": 4096000}
      ]
    }
  },

  "summary_metrics": {
    "repo_count": 12,
    "total_commits": 3275,
    "api_calls_used": {
      "total_calls": 54,
      "rest_calls": 53,
      "graphql_calls": 1
    }
  }
}
```

### Field Notes

- `repositories` contains ALL repos including forks. `is_fork: true` repos are excluded from feature calculations but present in raw JSON.
- Phase 1B fields (`readme_length_chars`, `has_ci_config`, `has_test_presence`, `recent_commit_messages`) are populated for original (non-fork) repos only via `enrich_repo_details()`.
- `commit_activity`: top 15 non-fork repos by `pushed_at`, 52-week weekly stats.
- `graphql_data.contribution_calendar.weeks`: 52 entries, 7 days each.
- `graphql_data.languages_breakdown`: top 100 repos by stars, top 10 languages each by size.

---

## 2. Known Bug Fixes Already Applied — DO NOT REINTRODUCE

### 2a. `doc_desc_coverage` — description field ONLY

**DO NOT** check `readme_length_chars > 0` or `r.get("language")` — those were the original bugs.  
`readme_length_chars` is owned by `doc_desc_depth_mean`. Counting it under `desc_coverage` is double-crediting.

**Exact current code** (`feature_engineering.py:481-486`):
```python
    # 1. desc_coverage: repos with non-empty description field ONLY
    has_desc_count = sum(
        1 for r in original
        if r.get("description") and len(str(r.get("description")).strip()) > 0
    )
    desc_coverage = _safe_div(has_desc_count, n_orig)
```

### 2b. `eng_ci_ratio` — real `.github/workflows` check

**DO NOT** infer CI from language names (YAML, Dockerfile, HCL, Shell). GitHub's language API does not surface YAML/Dockerfile for most repos.

**Exact current code** (`feature_engineering.py:553-568`):
```python
    ci_count = 0
    for r in original:
        if "has_ci_config" in r:
            if r.get("has_ci_config") is True:
                ci_count += 1
        else:
            # Fallback for legacy JSON where field is missing
            rname = r.get("name", "")
            langs = {e.get("language", "") for e in lb.get(rname, [])}
            langs.add(r.get("language") or "")
            if langs & DEVOPS_LANG_NAMES or "Dockerfile" in langs or "YAML" in langs:
                ci_count += 1
    ci_ratio = _safe_div(ci_count, n_orig)
```

The `"has_ci_config" in r` guard (not `r.get(...) is True`) ensures `False` values don't fall through to the legacy proxy. Only JSON without the key triggers the proxy.

### 2c. `eng_test_ratio` — real test directory/file presence

**DO NOT** count test ratio by language capability ("Python can have tests → count it"). That proxy does not correlate with whether tests actually exist.

Detection logic in `collect_profile.py:146-163`:
```python
test_dir_names     = {"tests", "test", "__tests__", "spec", "testing"}
test_file_patterns = ("test_", "_test.", ".test.", ".spec.")
# Checks item["type"] == "dir" and item["name"].lower() in test_dir_names
# OR item["type"] == "file" and any pattern in item["name"].lower()
```

Feature engineering fallback (`feature_engineering.py:567-581`):
```python
    test_count = 0
    for r in original:
        if "has_test_presence" in r:
            if r.get("has_test_presence") is True:
                test_count += 1
        else:
            # Fallback for legacy JSON where field is missing
            rname = r.get("name", "")
            langs = {e.get("language", "") for e in lb.get(rname, [])}
            langs.add(r.get("language") or "")
            if langs & TEST_CAPABLE_LANGUAGES and len(langs) >= 2:
                test_count += 1
    test_ratio = _safe_div(test_count, n_orig)
```

### 2d. Field naming — locked, do not rename

| Canonical Name | Measures | NOT acceptable |
|---|---|---|
| `eng_commit_message_quality` | Fraction of sampled commit subjects matching Conventional Commit / structured-prefix regex | ~~eng_commit_quality~~ |
| `eng_pr_discipline` | PR merge rate: `total_merged / total_opened` | — |
| `eng_review_participation` | Binary 1.0 if `review_participation_count > 0` | ~~eng_branch_discipline~~ — does not measure branch naming/strategy |

### 2e. Phase 1B fork exclusion — PARTIALLY FIXED, ONE LIVE BUG

The `enrich_repo_details()` loop (`collect_profile.py:120-125`):
```python
        for repo in repos:
            if repo.get("is_fork"):
                repo["readme_length_chars"] = 0
                repo["has_ci_config"] = False
                repo["recent_commit_messages"] = []
                continue         # exits here for forks
            # ... API calls follow
            repo["has_test_presence"] = has_tests  # NOT reached for forks!
```

**Bug:** `has_test_presence` is never set on fork repos — the `continue` fires before that assignment.

**Current mitigation:** The `"has_test_presence" in r` guard in feature engineering catches missing keys and falls back to the language proxy. Fork repos are excluded from `eng_test_ratio` via `original = [r for r in repos if not r.get("is_fork")]`, so this does not affect computed scores. But raw JSON is incomplete.

**Fix required (not yet applied) — one line addition:**
```python
            if repo.get("is_fork"):
                repo["readme_length_chars"] = 0
                repo["has_ci_config"] = False
                repo["has_test_presence"] = False   # ADD THIS
                repo["recent_commit_messages"] = []
                continue
```

---

## 3. Current DOC_WEIGHTS and ENG_WEIGHTS

Both runtime-asserted to sum to 1.0 at module load.

### DOC_WEIGHTS (`feature_engineering.py:129-135`)

```python
DOC_WEIGHTS = {
    "desc_coverage": 0.30,  # Fraction of original repos with non-empty description string
                             # Reads: repositories[i]["description"]
    "desc_depth":    0.25,  # Mean normalised README length: mean(min(readme_length_chars, 5000) / 5000)
                             # Reads: repositories[i]["readme_length_chars"]
    "bio":           0.15,  # 1.0 if user has non-empty bio, else 0.0
                             # Reads: user_info["bio"]
    "blog":          0.10,  # 1.0 if user has non-empty blog/website link, else 0.0
                             # Reads: user_info["blog"]
    "pinned":        0.20,  # Pinned repo showcase ratio: len(pinned_repositories) / 6
                             # Reads: graphql_data["pinned_repositories"]
}
# sum = 0.30 + 0.25 + 0.15 + 0.10 + 0.20 = 1.00
```

### ENG_WEIGHTS (`feature_engineering.py:139-145`)

```python
ENG_WEIGHTS = {
    "ci":                   0.30,  # Fraction of original repos with has_ci_config == True
                                    # Reads: repositories[i]["has_ci_config"]
    "test":                 0.25,  # Fraction of original repos with has_test_presence == True
                                    # Reads: repositories[i]["has_test_presence"]
    "commit_message":       0.20,  # Fraction of sampled commit subjects matching structured-prefix regex
                                    # Reads: repositories[i]["recent_commit_messages"]
    "pr_disc":              0.15,  # PR merge rate: total_merged / total_opened
                                    # Reads: pull_requests["total_merged"], pull_requests["total_opened"]
    "review_participation": 0.10,  # 1.0 if review_participation_count > 0, else 0.0
                                    # Reads: pull_requests["review_participation_count"]
}
# sum = 0.30 + 0.25 + 0.20 + 0.15 + 0.10 = 1.00
```

---

## 4. Full Worked Example — `torvalds`

**Fresh computation:** `python -m devlens.features.feature_engineering data/raw/torvalds.json` run 2026-08-06.  
torvalds: 12 public repos, 9 original (non-fork), 3 forks.

### `doc_score` Breakdown

| Component | Feature Key | Weight | Raw Value | Derivation | Weighted |
|---|---|---|---|---|---|
| Description Coverage | `doc_desc_coverage` | 0.30 | **1.0000** | 9/9 original repos have non-empty `description` | 0.3000 |
| README Depth | `doc_desc_depth_mean` | 0.25 | **0.6514** | mean of `min(readme_length_chars, 5000)/5000` across 9 repos | 0.1628 |
| Profile Bio | `doc_has_bio` | 0.15 | **0.0000** | `user_info["bio"]` is null | 0.0000 |
| Blog Link | `doc_has_blog` | 0.10 | **0.0000** | `user_info["blog"]` is empty | 0.0000 |
| Pinned Showcase | `doc_pinned_ratio` | 0.20 | **0.8333** | 5 pinned repos / 6 maximum | 0.1667 |
| **`doc_score`** | | **1.00** | | | **0.6295** |

`(0.30 × 1.0) + (0.25 × 0.6514) + (0.15 × 0) + (0.10 × 0) + (0.20 × 0.8333) = 0.6295`

### `eng_maturity_score` Breakdown

| Component | Feature Key | Weight | Raw Value | Derivation | Weighted |
|---|---|---|---|---|---|
| CI/CD Config | `eng_ci_ratio` | 0.30 | **0.1111** | 1/9 repos have `has_ci_config=True` (GuitarPedal only) | 0.0333 |
| Test Presence | `eng_test_ratio` | 0.25 | **0.1111** | 1/9 repos have `has_test_presence=True` (AudioNoise only) | 0.0278 |
| Commit Quality | `eng_commit_message_quality` | 0.20 | **0.1872** | 23/121 sampled commit subjects match structured-prefix regex | 0.0374 |
| PR Discipline | `eng_pr_discipline` | 0.15 | **0.8471** | 72 merged / 85 opened PRs | 0.1271 |
| Review Participation | `eng_review_participation` | 0.10 | **1.0000** | `review_participation_count=34 > 0` | 0.1000 |
| **`eng_maturity_score`** | | **1.00** | | | **0.3256** |

`(0.30 × 0.1111) + (0.25 × 0.1111) + (0.20 × 0.1872) + (0.15 × 0.8471) + (0.10 × 1.0) = 0.3256`

> **Important context:** Prior proxy-based calculation gave `eng_ci_ratio=0.8889`, `eng_maturity_score=0.6145`. Current empirical values (0.1111 each for CI and test) reflect real API checks. The linux repo has no `.github/workflows` because Linux kernel CI is external (kernel.org patchwork), not GitHub Actions.

### Commit Quality Regex (exact pattern)
```python
structured_pattern = re.compile(
    r'^(?:feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|Merge|Merge tag|\[.*\])',
    re.IGNORECASE
)
```
Applied across all `recent_commit_messages` lists from all original repos pooled together.

---

## 5. Data Collection Tiers

### Tier 1 — Consented Classmates (Active)
- **Count:** 194 usernames from `data/classmate_usernames.txt`
- **Source:** Cleaned from `data/classmate_links.txt` (207 raw lines) via `prep_usernames.py`
- **Depth:** Full Phase 1 + Phase 1B
  - Phase 1: user info, all repos, commit activity (top 15 non-fork by `pushed_at`), PR history, issue activity, GraphQL contributions + language breakdown + pinned repos
  - Phase 1B: per original repo — README length, CI config, test dir/file presence, up to 30 commit message subjects
- **CLI:** `python -m devlens.data_collection.collect_profile --batch data/classmate_usernames.txt`
- **Status:** 5-user test done. Full 194-user run not yet started.

### Tier 2 — Supplementary Pool (Not started)
- **Purpose:** Augment dataset only if Phase 5 clustering on Tier 1 alone shows silhouette score < 0.4 or unstable cluster assignments
- **Planned depth:** Phase 1 only — no README, CI, test-dir, or commit-message calls
- **Planned source:** GitHub Search API
- **Decision rule:** Do NOT build until Phase 5 clustering results are evaluated
- **Status:** Not started, no code written

---

## 6. Batch Collection Tooling

### CLI
```bash
# Full depth:
python -m devlens.data_collection.collect_profile --batch data/classmate_usernames.txt

# Test run (first N users):
python -m devlens.data_collection.collect_profile --batch data/classmate_usernames.txt --limit 5

# Reduced API usage (skips commit messages, saves ~30 calls/user):
python -m devlens.data_collection.collect_profile --batch data/classmate_usernames.txt --skip-commit-messages

# Single user:
python -m devlens.data_collection.collect_profile torvalds
```

### Resumability
- On startup reads existing `data/raw/{username}.json` and checks `collected_at` date
- Skips if date matches today's UTC date (string prefix `YYYY-MM-DD`)
- Safe to re-run after interruption; corrupt JSON re-fetched

### Rate Limit Handling
- `GitHubClient._handle_rate_limit()` reads `X-RateLimit-Remaining` from every response header
- Sleeps until `X-RateLimit-Reset` + 2s when remaining ≤ 1
- `client.last_rate_limit_remaining` updated on every REST call; shown in per-user progress output
- 5xx / network errors: exponential backoff, `max_retries=5` (Phase 1), `max_retries=2` (Phase 1B)

### `batch_summary.json` — actual output from 5-user test
```json
{
  "timestamp": "2026-08-06T08:41:47.927317Z",
  "input_file": "data/classmate_usernames.txt",
  "total_input_usernames": 194,
  "processed_count": 5,
  "skipped_count": 0,
  "succeeded_count": 4,
  "failed_count": 1,
  "failed_users": {
    "anbuchelvan-efx": "Could not retrieve user info for 'anbuchelvan-efx'."
  },
  "total_api_calls_consumed": 577,
  "elapsed_seconds": 324.18,
  "avg_repos_per_user": 27.0,
  "avg_api_calls_per_user": 144.0
}
```

### Rate Budget

| Metric | Value |
|---|---|
| Avg repos / user (5-user sample) | 27.0 |
| Avg API calls / user (5-user sample) | 144.0 |
| Estimated total for 194 users | **27,936 calls** |
| GitHub PAT limit | 5,000 / hr |
| **Estimated total time** | **~5.6 hours** |
| With `--skip-commit-messages` (saves ~30/user) | ~22,100 calls ≈ **~4.4 hours** |
| Users per safe hour | ~34 |

Batch is resumable — can split across multiple 1-hour sessions with the same command.

---

## 7. Outstanding Issues at Time of Handoff

### Issue 1: `anbuchelvan-efx` — collection failure
- **Error:** `GET /users/anbuchelvan-efx` returned non-200
- **Action:** Check `https://github.com/anbuchelvan-efx` — may be renamed, private, deleted, or typo. Update `data/classmate_usernames.txt` if account exists under different name.

### Issue 2: 2 unparseable entries in `classmate_links.txt`
- **Line 197:** `Suhitha M` — plain name, no GitHub handle. Manual lookup needed.
- **Line 200:** `Abisha_Rebekkal` — underscores invalid in GitHub usernames. Likely typo of `Abisha-Rebekkal21` (already in cleaned list at line 1). Confirm before adding to avoid duplicate.

### Issue 3: `has_test_presence` not set on fork repos — LIVE BUG (low impact)
- Fork repos exit `enrich_repo_details()` via `continue` before `has_test_presence` is assigned
- No impact on feature values (forks excluded from `eng_test_ratio` calculation)
- Raw JSON is incomplete — fork dicts have no `has_test_presence` key
- **One-line fix in `collect_profile.py:123`:** Add `repo["has_test_presence"] = False` inside the fork early-exit block before `continue`

### Issue 4: Full 194-user batch not yet run
- 5-user test only (4 succeeded, 1 failed)
- ~5.6 hours total estimated; run in resumable sessions respecting 5,000/hr limit

---

## 8. Pipeline Phases — Completed vs Pending

| Phase | Name | Status | Notes |
|---|---|---|---|
| 0 | Scaffold | DONE | venv, folder structure, requirements.txt, .env.example |
| 1 | Data Collection (REST + GraphQL) | DONE | `collect_profile.py`, `github_client.py` |
| 1B | Repository Enrichment (README/CI/test/commits) | DONE | `enrich_repo_details()` — 1 known bug (Issue 3) |
| 1C | Username Preparation | DONE | `prep_usernames.py`, `classmate_usernames.txt` |
| 1D | Batch Collection | DONE | batch mode in `collect_profile.py` |
| 2 | Resume Extractor | DONE | `devlens/data_collection/resume_extractor.py` |
| 3 | SQLite Schema + ORM | DONE | `models.py`, `session.py`, `repository.py`, `init_db.py` |
| 4 | Feature Engineering | PARTIAL | doc/eng scores ground-truth verified; activity/lang/repo/collab implemented but not proxy-audited |
| 5 | Clustering (K-Means / Archetypes) | NOT STARTED | Requires completed Tier 1 collection |
| 6 | SHAP Explainability | NOT STARTED | |
| 7 | Flask API | NOT STARTED | |
| 8 | Frontend Dashboard | NOT STARTED | |
| 9 | IEEE Paper Output | NOT STARTED | |

### Phase 4 — Feature Family Verification Status

| Family | Implemented | Ground-truth API audit |
|---|---|---|
| Activity (`activity_*`) | Yes | Not audited |
| Language/Stack (`lang_*`) | Yes | Not audited |
| Repository Quality (`repo_*`) | Yes | Not audited |
| Collaboration (`collab_*`) | Yes | Not audited |
| Documentation Score (`doc_score`) | Yes | **Audited** — verified against torvalds raw data |
| Engineering Maturity Score (`eng_maturity_score`) | Yes | **Audited** — verified against torvalds raw data |

---

## 9. Key File Map

| File | Purpose |
|---|---|
| `devlens/data_collection/github_client.py` | REST + GraphQL client, rate limiting, retry logic |
| `devlens/data_collection/collect_profile.py` | Single-user and batch profile collection |
| `devlens/data_collection/resume_extractor.py` | PDF/DOCX resume to GitHub username extraction |
| `devlens/data_collection/prep_usernames.py` | Batch URL/name to clean username preparation |
| `devlens/db/models.py` | SQLAlchemy ORM models |
| `devlens/db/session.py` | Engine and session factory (SQLite `devlens.db`) |
| `devlens/db/repository.py` | CRUD: upsert_developer, insert_snapshot, insert_features, scores |
| `devlens/db/init_db.py` | Creates all tables |
| `devlens/features/feature_engineering.py` | Full feature vector builder and CLI |
| `data/classmate_links.txt` | Original 207-line input (raw URLs/names from class) |
| `data/classmate_usernames.txt` | Cleaned 194-username output |
| `data/raw/{username}.json` | Per-user collected profile data |
| `data/raw/batch_summary.json` | Last batch run summary |
| `devlens.db` | SQLite database (gitignored) |

"""
DevLens Feature Engineering Module
===================================
Computes a flat numeric feature vector from a raw developer profile JSON
(produced by devlens.data_collection.collect_profile).

Feature Families
----------------
1. Activity          — commit frequency, consistency, cadence, calendar spread
2. Language/Stack    — language diversity, primary language, framework signals
3. Repository Quality — stars, forks, size distribution, fork ratio
4. Collaboration     — PR rates, issue closure, review participation
5. Documentation Score — composite score (see formula below)
6. Engineering Maturity Score — composite score (see formula below)

Documentation Score Formula (Section 3.4.1)
--------------------------------------------
Let N = number of original (non-fork) repositories.
Let R_i = 1 if repo i has a non-empty description, 0 otherwise.
Let P_i = 1 if repo i name contains 'readme' in any file (approximated by
          description length ≥ 50 characters as proxy), 0 otherwise.
Let D_i = description length of repo i (clamped to [0, 300] chars).
Let has_bio = 1 if GitHub bio is non-empty, 0 otherwise.
Let has_blog = 1 if GitHub blog/website URL is non-empty, 0 otherwise.
Let pinned_count = number of pinned repositories (0–6).

Component weights (sum to 1.0):
  w_desc    = 0.30  — proportion of repos with a description
  w_depth   = 0.25  — average normalised description depth (D_i / 300)
  w_bio     = 0.15  — developer bio completeness
  w_blog    = 0.10  — external documentation link presence
  w_pinned  = 0.20  — pinned repo showcase completeness (pinned_count / 6)

Formula:
  doc_score = (
      w_desc   * (sum(R_i) / max(N, 1))
    + w_depth  * (sum(min(D_i, 300) / 300 for all original repos) / max(N, 1))
    + w_bio    * has_bio
    + w_blog   * has_blog
    + w_pinned * (pinned_count / 6)
  )

Range: [0.0, 1.0].  Higher = better documentation posture.

Engineering Maturity Score Formula (Section 3.4.2)
----------------------------------------------------
Let N = number of original repositories.
Let CI_i = 1 if repo name, description, or language signals CI/CD tools
           (.github, Dockerfile, YAML, Actions keywords), 0 otherwise.
           Approximated via languages_breakdown containing Dockerfile or
           YAML language entries.
Let TEST_i = 1 if language breakdown includes a testing-associated language
             (Python with pytest patterns, Java, Go, Ruby, JavaScript with
             ≥2 language entries), used as a proxy for test file presence.
Let CONV_i  : commit message quality is not repo-level from REST API;
              approximated as 1.0 if PR merge rate ≥ 0.7 (indicating disciplined
              workflow) else proportional to merge_rate.
Let PR_DISC = merge_rate (PRs merged / PRs opened), range [0, 1].
Let BRANCH  = 1 if review_participation_count > 0 (indicates collaborative
              branching discipline), 0 otherwise.

Component weights (sum to 1.0):
  w_ci       = 0.30  — CI/CD toolchain presence
  w_test     = 0.25  — test infrastructure proxy
  w_commit   = 0.20  — commit message quality proxy (PR merge discipline)
  w_pr_disc  = 0.15  — PR workflow discipline (merge rate)
  w_branch   = 0.10  — branch/review discipline

Formula:
  ci_ratio   = sum(CI_i)   / max(N, 1)
  test_ratio = sum(TEST_i) / max(N, 1)
  commit_q   = min(PR_DISC / 0.7, 1.0)   # saturates at merge_rate = 0.7+
  pr_disc    = PR_DISC
  branch_d   = 1.0 if review_participation_count > 0 else 0.0

  eng_maturity_score = (
      w_ci     * ci_ratio
    + w_test   * test_ratio
    + w_commit * commit_q
    + w_pr_disc * pr_disc
    + w_branch * branch_d
  )

Range: [0.0, 1.0].  Higher = more mature engineering practices.

CLI Usage
---------
  python -m devlens.features.feature_engineering data/raw/torvalds.json
  python -m devlens.features.feature_engineering data/raw/torvalds.json --save-db
"""

import argparse
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Languages associated with frontend web development
FRONTEND_EXTENSIONS = {"tsx", "jsx", "ts", "css", "scss", "sass", "less", "html", "vue", "svelte"}
FRONTEND_LANGUAGES   = {"TypeScript", "JavaScript", "CSS", "HTML", "Vue", "Svelte"}

# Languages associated with DevOps/infrastructure
DEVOPS_LANGUAGES = {"Dockerfile", "HCL", "Shell", "YAML"}
DEVOPS_LANG_NAMES = {"Dockerfile", "HCL", "Shell", "Makefile"}

# Languages associated with ML/Data Science
ML_LANGUAGES = {"Jupyter Notebook", "Python", "R", "Julia"}
ML_MARKERS   = {"Jupyter Notebook"}

# Languages that often accompany test infrastructure
TEST_CAPABLE_LANGUAGES = {
    "Python", "Java", "Go", "Ruby", "Rust", "JavaScript", "TypeScript",
    "C#", "Kotlin", "Swift", "Scala"
}

# Documentation score weights (must sum to 1.0)
DOC_WEIGHTS = {
    "desc_coverage": 0.30,
    "desc_depth":    0.25,
    "bio":           0.15,
    "blog":          0.10,
    "pinned":        0.20,
}
assert abs(sum(DOC_WEIGHTS.values()) - 1.0) < 1e-9, "DOC_WEIGHTS must sum to 1.0"

# Engineering maturity score weights (must sum to 1.0)
ENG_WEIGHTS = {
    "ci":      0.30,
    "test":    0.25,
    "commit":  0.20,
    "pr_disc": 0.15,
    "branch":  0.10,
}
assert abs(sum(ENG_WEIGHTS.values()) - 1.0) < 1e-9, "ENG_WEIGHTS must sum to 1.0"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with zero-denominator guard."""
    return numerator / denominator if denominator > 0 else default


def _shannon_entropy(counts: List[float]) -> float:
    """Shannon entropy H = -sum(p_i * log2(p_i)) over a distribution of counts.
    Returns 0.0 if only one category or all counts are zero.
    """
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _coefficient_of_variation(values: List[float]) -> float:
    """CV = std / mean. Returns 0.0 if mean == 0."""
    if not values or len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    std  = arr.std()
    return _safe_div(std, mean, 0.0)


# ---------------------------------------------------------------------------
# Family 1: Activity features
# ---------------------------------------------------------------------------

def compute_activity_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """Compute commit activity features from 52-week repo-level stats
    and the GraphQL contribution calendar.

    Features
    --------
    activity_total_commits          : total contributions over 1 year (GraphQL calendar)
    activity_weekly_mean            : mean weekly commits across all repos in commit_activity
    activity_weekly_std             : std of weekly commit totals
    activity_weekly_cv              : coefficient of variation (consistency measure)
    activity_active_weeks_ratio     : fraction of 52 weeks with ≥1 commit
    activity_max_week               : peak weekly commits
    activity_calendar_days_active   : number of distinct calendar days with ≥1 contribution
    activity_longest_streak_days    : longest consecutive daily contribution streak
    """
    features: Dict[str, float] = {}

    # --- From GraphQL calendar (daily resolution) ---
    gql = raw.get("graphql_data") or {}
    cal = gql.get("contribution_calendar", {})
    weeks = cal.get("weeks", [])

    all_days: List[Dict[str, Any]] = []
    for week in weeks:
        all_days.extend(week.get("contributionDays", []))

    daily_counts = [d.get("contributionCount", 0) for d in all_days]
    features["activity_total_commits"]        = float(cal.get("totalContributions", 0))
    features["activity_calendar_days_active"] = float(sum(1 for c in daily_counts if c > 0))

    # Longest consecutive streak
    streak, max_streak = 0, 0
    for c in daily_counts:
        if c > 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    features["activity_longest_streak_days"] = float(max_streak)

    # --- From REST commit_activity (weekly resolution, per repo) ---
    commit_activity = raw.get("commit_activity", {})
    if commit_activity:
        # Aggregate totals per calendar week across all repos
        week_agg: Dict[int, int] = {}
        for repo_stats in commit_activity.values():
            if not isinstance(repo_stats, list):
                continue
            for entry in repo_stats:
                w = entry.get("week", 0)
                week_agg[w] = week_agg.get(w, 0) + entry.get("total", 0)

        weekly_totals = list(week_agg.values())
        n_weeks = max(len(weekly_totals), 52)

        features["activity_weekly_mean"]      = float(np.mean(weekly_totals)) if weekly_totals else 0.0
        features["activity_weekly_std"]       = float(np.std(weekly_totals))  if weekly_totals else 0.0
        features["activity_weekly_cv"]        = _coefficient_of_variation(weekly_totals)
        features["activity_active_weeks_ratio"] = _safe_div(
            sum(1 for w in weekly_totals if w > 0), n_weeks
        )
        features["activity_max_week"]         = float(max(weekly_totals)) if weekly_totals else 0.0
    else:
        # Fall back to calendar-derived weekly estimates
        cal_weekly = [
            sum(d.get("contributionCount", 0) for d in week.get("contributionDays", []))
            for week in weeks
        ]
        n_weeks = max(len(cal_weekly), 52)
        features["activity_weekly_mean"]        = float(np.mean(cal_weekly)) if cal_weekly else 0.0
        features["activity_weekly_std"]         = float(np.std(cal_weekly))  if cal_weekly else 0.0
        features["activity_weekly_cv"]          = _coefficient_of_variation(cal_weekly)
        features["activity_active_weeks_ratio"] = _safe_div(
            sum(1 for w in cal_weekly if w > 0), n_weeks
        )
        features["activity_max_week"]           = float(max(cal_weekly)) if cal_weekly else 0.0

    return features


# ---------------------------------------------------------------------------
# Family 2: Language / Stack features
# ---------------------------------------------------------------------------

def compute_language_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """Compute language diversity and framework signal features.

    Features
    --------
    lang_diversity_entropy          : Shannon entropy over total LOC per language (nats base 2)
    lang_primary_is_c               : 1.0 if primary language is C or C++
    lang_primary_is_python          : 1.0 if primary language is Python
    lang_primary_is_javascript      : 1.0 if primary language is JS/TS
    lang_primary_is_java_kotlin     : 1.0 if primary language is Java or Kotlin
    lang_primary_is_go              : 1.0 if primary language is Go
    lang_primary_is_rust            : 1.0 if primary language is Rust
    lang_unique_count               : number of distinct languages used across repos
    lang_frontend_signal            : fraction of repos with frontend language indicators
    lang_devops_signal              : fraction of repos with DevOps language indicators
    lang_ml_signal                  : fraction of repos with ML language indicators
    """
    features: Dict[str, float] = {}
    repos = raw.get("repositories", [])
    gql   = raw.get("graphql_data") or {}
    lang_breakdown = gql.get("languages_breakdown", {})

    # Aggregate total bytes per language across all repos
    lang_bytes: Dict[str, int] = {}
    for repo_langs in lang_breakdown.values():
        for entry in repo_langs:
            lang = entry.get("language", "Unknown")
            lang_bytes[lang] = lang_bytes.get(lang, 0) + entry.get("size_bytes", 0)

    if lang_bytes:
        features["lang_diversity_entropy"] = _shannon_entropy(list(lang_bytes.values()))
        primary = max(lang_bytes, key=lambda k: lang_bytes[k]) if lang_bytes else None
        features["lang_unique_count"]      = float(len(lang_bytes))
    else:
        # Fall back to per-repo primary language from REST
        lang_counts: Dict[str, int] = {}
        for r in repos:
            lang = r.get("language") or "Unknown"
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        features["lang_diversity_entropy"] = _shannon_entropy(list(lang_counts.values()))
        features["lang_unique_count"]      = float(len(lang_counts))
        primary = max(lang_counts, key=lambda k: lang_counts[k]) if lang_counts else None

    # Primary language binary flags
    primary_norm = (primary or "").lower()
    features["lang_primary_is_c"]           = 1.0 if primary_norm in ("c", "c++", "c/c++") else 0.0
    features["lang_primary_is_python"]      = 1.0 if primary_norm == "python" else 0.0
    features["lang_primary_is_javascript"]  = 1.0 if primary_norm in ("javascript", "typescript") else 0.0
    features["lang_primary_is_java_kotlin"] = 1.0 if primary_norm in ("java", "kotlin") else 0.0
    features["lang_primary_is_go"]          = 1.0 if primary_norm == "go" else 0.0
    features["lang_primary_is_rust"]        = 1.0 if primary_norm == "rust" else 0.0

    # Framework signals per repo
    original_repos = [r for r in repos if not r.get("is_fork")]
    n_orig = max(len(original_repos), 1)

    frontend_count, devops_count, ml_count = 0, 0, 0
    for r in original_repos:
        rname  = r.get("name", "")
        langs  = {e.get("language") for e in lang_breakdown.get(rname, [])}
        # also include repo primary language
        langs.add(r.get("language") or "")

        if langs & FRONTEND_LANGUAGES:
            frontend_count += 1
        if langs & DEVOPS_LANG_NAMES:
            devops_count += 1
        if langs & ML_MARKERS:
            ml_count += 1

    features["lang_frontend_signal"] = _safe_div(frontend_count, n_orig)
    features["lang_devops_signal"]   = _safe_div(devops_count,   n_orig)
    features["lang_ml_signal"]       = _safe_div(ml_count,       n_orig)

    return features


# ---------------------------------------------------------------------------
# Family 3: Repository Quality features
# ---------------------------------------------------------------------------

def compute_repo_quality_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """Compute repository quality and portfolio composition features.

    Features
    --------
    repo_total_count                : total public repositories
    repo_original_count             : non-fork repositories
    repo_fork_ratio                 : forks / total repos
    repo_total_stars                : sum of stars across all repos
    repo_total_forks_received       : sum of forks received
    repo_mean_stars                 : mean stars per original repo
    repo_max_stars                  : max stars on any single repo
    repo_mean_size_kb               : mean repo size in KB
    repo_std_size_kb                : std of repo sizes (portfolio breadth indicator)
    repo_years_active               : span from earliest created_at to most recent pushed_at
    """
    features: Dict[str, float] = {}
    repos = raw.get("repositories", [])

    if not repos:
        return {k: 0.0 for k in [
            "repo_total_count", "repo_original_count", "repo_fork_ratio",
            "repo_total_stars", "repo_total_forks_received", "repo_mean_stars",
            "repo_max_stars", "repo_mean_size_kb", "repo_std_size_kb", "repo_years_active"
        ]}

    original = [r for r in repos if not r.get("is_fork")]
    n_total  = len(repos)
    n_orig   = len(original)

    stars  = [r.get("stars",  0) or 0 for r in original]
    forks  = [r.get("forks",  0) or 0 for r in original]
    sizes  = [r.get("size",   0) or 0 for r in repos]

    features["repo_total_count"]          = float(n_total)
    features["repo_original_count"]       = float(n_orig)
    features["repo_fork_ratio"]           = _safe_div(n_total - n_orig, n_total)
    features["repo_total_stars"]          = float(sum(stars))
    features["repo_total_forks_received"] = float(sum(forks))
    features["repo_mean_stars"]           = float(np.mean(stars)) if stars else 0.0
    features["repo_max_stars"]            = float(max(stars))     if stars else 0.0
    features["repo_mean_size_kb"]         = float(np.mean(sizes)) if sizes else 0.0
    features["repo_std_size_kb"]          = float(np.std(sizes))  if sizes else 0.0

    # Active years span
    dates = []
    for r in repos:
        for field in ("created_at", "pushed_at"):
            v = r.get(field)
            if v:
                try:
                    from datetime import datetime
                    dates.append(datetime.fromisoformat(v.replace("Z", "+00:00")))
                except ValueError:
                    pass
    if len(dates) >= 2:
        delta_days = (max(dates) - min(dates)).days
        features["repo_years_active"] = delta_days / 365.25
    else:
        features["repo_years_active"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Family 4: Collaboration features
# ---------------------------------------------------------------------------

def compute_collaboration_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """Compute collaboration signal features from PR, issue and review data.

    Features
    --------
    collab_pr_opened                : total PRs opened
    collab_pr_merged                : total PRs merged
    collab_pr_merge_rate            : merged / opened (PR discipline)
    collab_review_count             : review participation events
    collab_review_rate_per_pr       : reviews / PRs opened (relative engagement)
    collab_issues_opened            : total issues opened
    collab_issues_closed            : total issues closed
    collab_issue_close_rate         : closed / opened (issue resolution discipline)
    collab_issues_commented         : total issue comments
    """
    features: Dict[str, float] = {}
    prs    = raw.get("pull_requests", {}) or {}
    issues = raw.get("issues",        {}) or {}

    pr_opened  = prs.get("total_opened",              0) or 0
    pr_merged  = prs.get("total_merged",              0) or 0
    pr_reviews = prs.get("review_participation_count", 0) or 0
    iss_opened = issues.get("total_opened", 0) or 0
    iss_closed = issues.get("total_closed", 0) or 0
    iss_comm   = issues.get("total_commented", 0) or 0

    features["collab_pr_opened"]            = float(pr_opened)
    features["collab_pr_merged"]            = float(pr_merged)
    features["collab_pr_merge_rate"]        = _safe_div(pr_merged,  pr_opened)
    features["collab_review_count"]         = float(pr_reviews)
    features["collab_review_rate_per_pr"]   = _safe_div(pr_reviews, pr_opened)
    features["collab_issues_opened"]        = float(iss_opened)
    features["collab_issues_closed"]        = float(iss_closed)
    features["collab_issue_close_rate"]     = _safe_div(iss_closed, iss_opened)
    features["collab_issues_commented"]     = float(iss_comm)

    return features


# ---------------------------------------------------------------------------
# Family 5: Documentation Score
# ---------------------------------------------------------------------------

def compute_documentation_score(raw: Dict[str, Any]) -> Dict[str, float]:
    """Compute the Documentation Score and its sub-components.

    See module docstring for the full citable formula (Section 3.4.1).

    Features
    --------
    doc_score                   : composite Documentation Score ∈ [0, 1]
    doc_desc_coverage           : proportion of original repos with a description
    doc_desc_depth_mean         : mean normalised description depth (len / 300)
    doc_has_bio                 : 1.0 if user bio is non-empty
    doc_has_blog                : 1.0 if user blog/website is set
    doc_pinned_ratio            : pinned_count / 6 (max pinnable)
    """
    features: Dict[str, float] = {}
    repos     = raw.get("repositories", []) or []
    user_info = raw.get("user_info",    {}) or {}
    gql       = raw.get("graphql_data") or {}

    original  = [r for r in repos if not r.get("is_fork")]
    n_orig    = max(len(original), 1)

    # desc_coverage: approximated via repo description from user_info
    # (REST repo list doesn't include 'description'; we use name/language as
    #  a presence proxy — for repos with stars>0 as a documentation-effort signal)
    # Best available proxy: starred repos with a defined primary language
    repos_with_lang  = sum(1 for r in original if r.get("language"))
    desc_coverage    = _safe_div(repos_with_lang, n_orig)

    # desc_depth: use GraphQL pinned repo descriptions as the richest source
    pinned = gql.get("pinned_repositories", []) or []
    all_descs = [p.get("description") or "" for p in pinned]
    # Also include description proxy from REST (language fields as tokens)
    desc_depths = [min(len(d), 300) / 300.0 for d in all_descs]
    desc_depth_mean = float(np.mean(desc_depths)) if desc_depths else 0.0

    has_bio  = 1.0 if user_info.get("bio")  else 0.0
    has_blog = 1.0 if user_info.get("blog") else 0.0

    pinned_count = len(pinned)
    pinned_ratio = min(pinned_count / 6.0, 1.0)

    doc_score = (
        DOC_WEIGHTS["desc_coverage"] * desc_coverage
      + DOC_WEIGHTS["desc_depth"]    * desc_depth_mean
      + DOC_WEIGHTS["bio"]           * has_bio
      + DOC_WEIGHTS["blog"]          * has_blog
      + DOC_WEIGHTS["pinned"]        * pinned_ratio
    )

    features["doc_score"]             = round(doc_score, 6)
    features["doc_desc_coverage"]     = round(desc_coverage, 6)
    features["doc_desc_depth_mean"]   = round(desc_depth_mean, 6)
    features["doc_has_bio"]           = has_bio
    features["doc_has_blog"]          = has_blog
    features["doc_pinned_ratio"]      = round(pinned_ratio, 6)
    features["doc_pinned_count"]      = float(pinned_count)

    return features


# ---------------------------------------------------------------------------
# Family 6: Engineering Maturity Score
# ---------------------------------------------------------------------------

def compute_engineering_maturity_score(raw: Dict[str, Any]) -> Dict[str, float]:
    """Compute the Engineering Maturity Score and its sub-components.

    See module docstring for the full citable formula (Section 3.4.2).

    Features
    --------
    eng_maturity_score          : composite Engineering Maturity Score ∈ [0, 1]
    eng_ci_ratio                : fraction of original repos with CI/CD language signals
    eng_test_ratio              : fraction of original repos with test-capable languages
    eng_commit_quality          : PR merge discipline proxy ∈ [0, 1]
    eng_pr_discipline           : PR merge rate (merged / opened)
    eng_branch_discipline       : 1.0 if review_participation > 0
    """
    features: Dict[str, float] = {}
    repos   = raw.get("repositories", []) or []
    prs     = raw.get("pull_requests", {}) or {}
    gql     = raw.get("graphql_data")  or {}
    lb      = gql.get("languages_breakdown", {}) or {}

    original = [r for r in repos if not r.get("is_fork")]
    n_orig   = max(len(original), 1)

    ci_count   = 0
    test_count = 0
    for r in original:
        rname = r.get("name", "")
        langs = {e.get("language", "") for e in lb.get(rname, [])}
        langs.add(r.get("language") or "")

        if langs & DEVOPS_LANG_NAMES or "Dockerfile" in langs or "YAML" in langs:
            ci_count += 1
        if langs & TEST_CAPABLE_LANGUAGES and len(langs) >= 2:
            test_count += 1

    ci_ratio   = _safe_div(ci_count,   n_orig)
    test_ratio = _safe_div(test_count, n_orig)

    pr_opened  = prs.get("total_opened",              0) or 0
    pr_merged  = prs.get("total_merged",              0) or 0
    pr_reviews = prs.get("review_participation_count", 0) or 0

    pr_disc    = _safe_div(pr_merged, pr_opened)
    commit_q   = min(_safe_div(pr_disc, 0.7), 1.0)   # saturates at 70% merge rate
    branch_d   = 1.0 if pr_reviews > 0 else 0.0

    eng_score = (
        ENG_WEIGHTS["ci"]      * ci_ratio
      + ENG_WEIGHTS["test"]    * test_ratio
      + ENG_WEIGHTS["commit"]  * commit_q
      + ENG_WEIGHTS["pr_disc"] * pr_disc
      + ENG_WEIGHTS["branch"]  * branch_d
    )

    features["eng_maturity_score"]    = round(eng_score,    6)
    features["eng_ci_ratio"]          = round(ci_ratio,     6)
    features["eng_test_ratio"]        = round(test_ratio,   6)
    features["eng_commit_quality"]    = round(commit_q,     6)
    features["eng_pr_discipline"]     = round(pr_disc,      6)
    features["eng_branch_discipline"] = branch_d

    return features


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_feature_vector(raw_json: Dict[str, Any]) -> Dict[str, float]:
    """Build a complete, flat numeric feature vector from a raw profile dict.

    Calls all six feature families in order and merges results into a
    single dict. Keys are namespaced by family prefix (activity_, lang_,
    repo_, collab_, doc_, eng_).

    Args:
        raw_json: Dict loaded from data/raw/{username}.json

    Returns:
        Flat dict mapping feature_name -> float value.
    """
    vector: Dict[str, float] = {}
    username = raw_json.get("username", "unknown")

    families = [
        ("Activity",              compute_activity_features),
        ("Language/Stack",        compute_language_features),
        ("Repository Quality",    compute_repo_quality_features),
        ("Collaboration",         compute_collaboration_features),
        ("Documentation Score",   compute_documentation_score),
        ("Engineering Maturity",  compute_engineering_maturity_score),
    ]

    for name, fn in families:
        try:
            result = fn(raw_json)
            vector.update(result)
            logger.debug(f"[{username}] {name}: {len(result)} features computed")
        except Exception as e:
            logger.error(f"[{username}] Error computing {name} features: {e}")
            raise

    logger.info(f"[{username}] Total features computed: {len(vector)}")
    return vector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_vector(vector: Dict[str, float], username: str) -> None:
    """Pretty-print the feature vector grouped by family prefix."""
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  DEV LENS - FEATURE VECTOR: {username}  ({len(vector)} features)")
    print(sep)

    groups = {}
    for k, v in vector.items():
        prefix = k.split("_")[0]
        groups.setdefault(prefix, {})[k] = v

    for prefix, feats in groups.items():
        print(f"\n  [{prefix.upper()}]")
        for fname, fval in feats.items():
            bar = "#" * min(int(fval * 20), 20) if 0.0 <= fval <= 1.0 else ""
            print(f"    {fname:<40} {fval:>12.4f}  {bar}")

    print(f"\n{sep}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute DevLens feature vector from a raw JSON profile."
    )
    parser.add_argument("json_path", type=str, help="Path to raw JSON file (data/raw/<username>.json)")
    parser.add_argument("--save-db", action="store_true", help="Write features to the database via the repository")
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"ERROR: File not found: {args.json_path}")
        raise SystemExit(1)

    with open(args.json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    username = raw.get("username", os.path.splitext(os.path.basename(args.json_path))[0])
    vector   = build_feature_vector(raw)
    _print_vector(vector, username)

    if args.save_db:
        from devlens.db.session import get_session, init_db, get_engine
        from devlens.db import repository as repo
        from datetime import datetime

        engine = init_db(get_engine())
        with get_session(engine) as session:
            dev  = repo.upsert_developer(session, username=username,
                                         resume_source=raw.get("collected_at"))
            snap = repo.insert_snapshot(session, developer_id=dev.id,
                                        raw_json_path=args.json_path,
                                        collected_at=datetime.utcnow())
            count = repo.insert_features(session, snap.id, vector)

            # Persist composite scores separately into score tables
            doc_score = vector.get("doc_score", 0.0)
            eng_score = vector.get("eng_maturity_score", 0.0)

            repo.upsert_documentation_score(
                session, snap.id, score=doc_score,
                components={k: v for k, v in vector.items() if k.startswith("doc_")}
            )
            repo.upsert_engineering_maturity_score(
                session, snap.id, score=eng_score,
                components={k: v for k, v in vector.items() if k.startswith("eng_")}
            )

        print(f"  [OK] Saved {count} features to DB for '{username}' (snapshot written)\n")


if __name__ == "__main__":
    main()

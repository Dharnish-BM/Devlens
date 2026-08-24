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
Let CI_i = 1 if repo has_ci_config == True, 0 otherwise.
Let TEST_i = 1 if repo has_test_presence == True, 0 otherwise.
Let CONV_i = ratio of structured/conventional commit message subjects.
Let PR_DISC = merge_rate (PRs merged / PRs opened), range [0, 1].
Let BRANCH = 1 if review_participation_count > 0, 0 otherwise.

Component weights (sum to 1.0):
  w_ci       = 0.30  — CI/CD toolchain presence
  w_test     = 0.25  — test infrastructure presence
  w_commit   = 0.20  — commit message quality
  w_pr_disc  = 0.15  — PR workflow discipline (merge rate)
  w_branch   = 0.10  — branch/review discipline

Formula:
  ci_ratio   = sum(CI_i) / max(N, 1)
  test_ratio = sum(TEST_i) / max(N, 1)
  commit_q   = ratio of structured commit subjects in recent_commit_messages
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
  Single user:
    python -m devlens.features.feature_engineering data/raw/torvalds.json
    python -m devlens.features.feature_engineering data/raw/torvalds.json --save-db

  Batch mode:
    python -m devlens.features.feature_engineering --batch data/raw/
"""

import argparse
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRONTEND_EXTENSIONS = {"tsx", "jsx", "ts", "css", "scss", "sass", "less", "html", "vue", "svelte"}
FRONTEND_LANGUAGES   = {"TypeScript", "JavaScript", "CSS", "HTML", "Vue", "Svelte"}

DEVOPS_LANGUAGES = {"Dockerfile", "HCL", "Shell", "YAML"}
DEVOPS_LANG_NAMES = {"Dockerfile", "HCL", "Shell", "Makefile"}

ML_LANGUAGES = {"Jupyter Notebook", "Python", "R", "Julia"}
ML_MARKERS   = {"Jupyter Notebook"}

MOBILE_LANG_NAMES = {"Swift", "Dart", "Kotlin", "Objective-C"}

TEST_CAPABLE_LANGUAGES = {
    "Python", "Java", "Go", "Ruby", "Rust", "JavaScript", "TypeScript",
    "C#", "Kotlin", "Swift", "Scala"
}

DOC_WEIGHTS = {
    "desc_coverage": 0.30,
    "desc_depth":    0.25,
    "bio":           0.15,
    "blog":          0.10,
    "pinned":        0.20,
}
assert abs(sum(DOC_WEIGHTS.values()) - 1.0) < 1e-9, "DOC_WEIGHTS must sum to 1.0"

ENG_WEIGHTS = {
    "ci":                   0.30,
    "test":                 0.25,
    "commit_message":       0.20,
    "pr_disc":              0.15,
    "review_participation": 0.10,
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
    """
    features: Dict[str, float] = {}

    gql = raw.get("graphql_data") or {}
    if not isinstance(gql, dict):
        gql = {}
    cal = gql.get("contribution_calendar") or {}
    if not isinstance(cal, dict):
        cal = {}
    weeks = cal.get("weeks") or []
    if not isinstance(weeks, list):
        weeks = []

    all_days: List[Dict[str, Any]] = []
    for week in weeks:
        if isinstance(week, dict):
            all_days.extend(week.get("contributionDays") or [])

    daily_counts = [d.get("contributionCount", 0) or 0 for d in all_days if isinstance(d, dict)]
    features["activity_total_commits"]        = float(cal.get("totalContributions", 0) or 0)
    features["activity_calendar_days_active"] = float(sum(1 for c in daily_counts if c > 0))

    streak, max_streak = 0, 0
    for c in daily_counts:
        if c > 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    features["activity_longest_streak_days"] = float(max_streak)

    commit_activity = raw.get("commit_activity") or {}
    if isinstance(commit_activity, dict) and commit_activity:
        week_agg: Dict[int, int] = {}
        for repo_stats in commit_activity.values():
            if not isinstance(repo_stats, list):
                continue
            for entry in repo_stats:
                if isinstance(entry, dict):
                    w = entry.get("week", 0)
                    week_agg[w] = week_agg.get(w, 0) + (entry.get("total", 0) or 0)

        weekly_totals = list(week_agg.values())
        n_weeks = max(len(weekly_totals), 52)

        features["activity_weekly_mean"]        = float(np.mean(weekly_totals)) if weekly_totals else 0.0
        features["activity_weekly_std"]         = float(np.std(weekly_totals))  if weekly_totals else 0.0
        features["activity_weekly_cv"]          = _coefficient_of_variation(weekly_totals)
        features["activity_active_weeks_ratio"] = _safe_div(
            sum(1 for w in weekly_totals if w > 0), n_weeks
        )
        features["activity_max_week"]           = float(max(weekly_totals)) if weekly_totals else 0.0
    else:
        cal_weekly = [
            sum((d.get("contributionCount", 0) or 0) for d in week.get("contributionDays") or [] if isinstance(d, dict))
            for week in weeks if isinstance(week, dict)
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
    """Compute language diversity and framework signal features."""
    features: Dict[str, float] = {}
    repos = raw.get("repositories") or []
    if not isinstance(repos, list):
        repos = []
    gql = raw.get("graphql_data") or {}
    if not isinstance(gql, dict):
        gql = {}
    lang_breakdown = gql.get("languages_breakdown") or {}
    if not isinstance(lang_breakdown, dict):
        lang_breakdown = {}

    lang_bytes: Dict[str, int] = {}
    for repo_langs in lang_breakdown.values():
        if not isinstance(repo_langs, list):
            continue
        for entry in repo_langs:
            if isinstance(entry, dict):
                lang = entry.get("language", "Unknown")
                lang_bytes[lang] = lang_bytes.get(lang, 0) + (entry.get("size_bytes", 0) or 0)

    if lang_bytes:
        features["lang_diversity_entropy"] = _shannon_entropy(list(lang_bytes.values()))
        primary = max(lang_bytes, key=lambda k: lang_bytes[k]) if lang_bytes else None
        features["lang_unique_count"]      = float(len(lang_bytes))
    else:
        lang_counts: Dict[str, int] = {}
        for r in repos:
            if isinstance(r, dict):
                lang = r.get("language") or "Unknown"
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        features["lang_diversity_entropy"] = _shannon_entropy(list(lang_counts.values()))
        features["lang_unique_count"]      = float(len(lang_counts))
        primary = max(lang_counts, key=lambda k: lang_counts[k]) if lang_counts else None

    primary_norm = (primary or "").lower()
    features["lang_primary_is_c"]           = 1.0 if primary_norm in ("c", "c++", "c/c++") else 0.0
    features["lang_primary_is_python"]      = 1.0 if primary_norm == "python" else 0.0
    features["lang_primary_is_javascript"]  = 1.0 if primary_norm in ("javascript", "typescript") else 0.0
    features["lang_primary_is_java_kotlin"] = 1.0 if primary_norm in ("java", "kotlin") else 0.0
    features["lang_primary_is_go"]          = 1.0 if primary_norm == "go" else 0.0
    features["lang_primary_is_rust"]        = 1.0 if primary_norm == "rust" else 0.0

    original_repos = [r for r in repos if isinstance(r, dict) and not r.get("is_fork")]
    n_orig = len(original_repos)

    frontend_count, devops_count, ml_count, mobile_count = 0, 0, 0, 0
    if original_repos:
        for r in original_repos:
            rname = r.get("name", "")
            raw_repo_langs = lang_breakdown.get(rname, []) if isinstance(lang_breakdown, dict) else []
            if not isinstance(raw_repo_langs, list):
                raw_repo_langs = []
            langs = {e.get("language") for e in raw_repo_langs if isinstance(e, dict) and e.get("language")}
            langs.add(r.get("language") or "")

            if langs & FRONTEND_LANGUAGES:
                frontend_count += 1
            if langs & DEVOPS_LANG_NAMES:
                devops_count += 1
            if langs & ML_MARKERS:
                ml_count += 1
            if langs & MOBILE_LANG_NAMES:
                mobile_count += 1

        features["lang_frontend_signal"] = _safe_div(frontend_count, n_orig)
        features["lang_devops_signal"]   = _safe_div(devops_count,   n_orig)
        features["lang_ml_signal"]       = _safe_div(ml_count,       n_orig)
        features["lang_mobile_signal"]   = _safe_div(mobile_count,   n_orig)
    else:
        # Defined default: 0.0 (no original work to assess)
        features["lang_frontend_signal"] = 0.0
        features["lang_devops_signal"]   = 0.0
        features["lang_ml_signal"]       = 0.0
        features["lang_mobile_signal"]   = 0.0

    return features


# ---------------------------------------------------------------------------
# Family 3: Repository Quality features
# ---------------------------------------------------------------------------

def compute_repo_quality_features(raw: Dict[str, Any]) -> Dict[str, float]:
    """Compute repository quality and portfolio composition features."""
    features: Dict[str, float] = {}
    repos = raw.get("repositories") or []
    if not isinstance(repos, list):
        repos = []

    if not repos:
        return {k: 0.0 for k in [
            "repo_total_count", "repo_original_count", "repo_fork_ratio",
            "repo_total_stars", "repo_total_forks_received", "repo_mean_stars",
            "repo_max_stars", "repo_mean_size_kb", "repo_std_size_kb", "repo_years_active"
        ]}

    original = [r for r in repos if isinstance(r, dict) and not r.get("is_fork")]
    n_total  = len(repos)
    n_orig   = len(original)

    stars  = [r.get("stars",  0) or 0 for r in original]
    forks  = [r.get("forks",  0) or 0 for r in original]
    sizes  = [r.get("size",   0) or 0 for r in original]

    features["repo_total_count"]          = float(n_total)
    features["repo_original_count"]       = float(n_orig)
    features["repo_fork_ratio"]           = _safe_div(n_total - n_orig, n_total)
    features["repo_total_stars"]          = float(sum(stars))
    features["repo_total_forks_received"] = float(sum(forks))
    features["repo_mean_stars"]           = float(np.mean(stars)) if stars else 0.0
    features["repo_max_stars"]            = float(max(stars))     if stars else 0.0
    features["repo_mean_size_kb"]         = float(np.mean(sizes)) if sizes else 0.0
    features["repo_std_size_kb"]          = float(np.std(sizes))  if sizes else 0.0

    dates = []
    for r in original:
        if isinstance(r, dict):
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
    """Compute collaboration signal features from PR, issue and review data."""
    features: Dict[str, float] = {}
    prs    = raw.get("pull_requests") or {}
    if not isinstance(prs, dict):
        prs = {}
    issues = raw.get("issues") or {}
    if not isinstance(issues, dict):
        issues = {}

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
    """Compute the Documentation Score and its sub-components."""
    features: Dict[str, float] = {}
    repos     = raw.get("repositories") or []
    if not isinstance(repos, list):
        repos = []
    user_info = raw.get("user_info") or {}
    if not isinstance(user_info, dict):
        user_info = {}
    gql       = raw.get("graphql_data") or {}
    if not isinstance(gql, dict):
        gql = {}

    original = [r for r in repos if isinstance(r, dict) and not r.get("is_fork")]

    if not original:
        desc_coverage = 0.0
        desc_depth_mean = 0.0
    else:
        n_orig = len(original)
        has_desc_count = sum(
            1 for r in original
            if r.get("description") and len(str(r.get("description")).strip()) > 0
        )
        desc_coverage = _safe_div(has_desc_count, n_orig)

        readme_lens = [r.get("readme_length_chars", 0) or 0 for r in original]
        if any(l > 0 for l in readme_lens):
            desc_depths = [min(l, 5000) / 5000.0 for l in readme_lens]
        else:
            pinned = gql.get("pinned_repositories") or []
            if not isinstance(pinned, list):
                pinned = []
            all_descs = [p.get("description") or "" for p in pinned if isinstance(p, dict)]
            desc_depths = [min(len(d), 300) / 300.0 for d in all_descs] if all_descs else [0.0]

        desc_depth_mean = float(np.mean(desc_depths)) if desc_depths else 0.0

    has_bio  = 1.0 if user_info.get("bio")  else 0.0
    has_blog = 1.0 if user_info.get("blog") else 0.0

    pinned = gql.get("pinned_repositories") or []
    if not isinstance(pinned, list):
        pinned = []
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
    """Compute the Engineering Maturity Score and its sub-components."""
    features: Dict[str, float] = {}
    repos   = raw.get("repositories") or []
    if not isinstance(repos, list):
        repos = []
    prs     = raw.get("pull_requests") or {}
    if not isinstance(prs, dict):
        prs = {}
    gql     = raw.get("graphql_data") or {}
    if not isinstance(gql, dict):
        gql = {}
    lb      = gql.get("languages_breakdown") or {}
    if not isinstance(lb, dict):
        lb = {}

    original = [r for r in repos if isinstance(r, dict) and not r.get("is_fork")]

    if not original:
        ci_ratio = 0.0
        test_ratio = 0.0
        commit_q = 0.0
    else:
        n_orig = len(original)

        ci_count = 0
        for r in original:
            if "has_ci_config" in r:
                if r.get("has_ci_config") is True:
                    ci_count += 1
            else:
                rname = r.get("name", "")
                raw_lb = lb.get(rname, []) if isinstance(lb, dict) else []
                if not isinstance(raw_lb, list):
                    raw_lb = []
                langs = {e.get("language", "") for e in raw_lb if isinstance(e, dict)}
                langs.add(r.get("language") or "")
                if langs & DEVOPS_LANG_NAMES or "Dockerfile" in langs or "YAML" in langs:
                    ci_count += 1
        ci_ratio = _safe_div(ci_count, n_orig)

        test_count = 0
        for r in original:
            if "has_test_presence" in r:
                if r.get("has_test_presence") is True:
                    test_count += 1
            else:
                rname = r.get("name", "")
                raw_lb = lb.get(rname, []) if isinstance(lb, dict) else []
                if not isinstance(raw_lb, list):
                    raw_lb = []
                langs = {e.get("language", "") for e in raw_lb if isinstance(e, dict)}
                langs.add(r.get("language") or "")
                if langs & TEST_CAPABLE_LANGUAGES and len(langs) >= 2:
                    test_count += 1
        test_ratio = _safe_div(test_count, n_orig)

        all_commit_msgs = []
        for r in original:
            msgs = r.get("recent_commit_messages")
            if isinstance(msgs, list):
                all_commit_msgs.extend(msgs)

        if all_commit_msgs:
            structured_pattern = re.compile(
                r'^(?:feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|Merge|Merge tag|\[.*\])',
                re.IGNORECASE
            )
            structured_count = sum(1 for m in all_commit_msgs if isinstance(m, str) and structured_pattern.match(m))
            commit_q = _safe_div(structured_count, len(all_commit_msgs))
        else:
            pr_opened = prs.get("total_opened", 0) or 0
            pr_merged = prs.get("total_merged", 0) or 0
            pr_disc   = _safe_div(pr_merged, pr_opened)
            commit_q  = min(_safe_div(pr_disc, 0.7), 1.0)

    pr_opened  = prs.get("total_opened", 0) or 0
    pr_merged  = prs.get("total_merged", 0) or 0
    pr_reviews = prs.get("review_participation_count", 0) or 0

    pr_disc  = _safe_div(pr_merged, pr_opened)
    review_p = 1.0 if pr_reviews > 0 else 0.0

    eng_score = (
        ENG_WEIGHTS["ci"]                   * ci_ratio
      + ENG_WEIGHTS["test"]                 * test_ratio
      + ENG_WEIGHTS["commit_message"]       * commit_q
      + ENG_WEIGHTS["pr_disc"]              * pr_disc
      + ENG_WEIGHTS["review_participation"] * review_p
    )

    features["eng_maturity_score"]        = round(eng_score,    6)
    features["eng_ci_ratio"]              = round(ci_ratio,     6)
    features["eng_test_ratio"]            = round(test_ratio,   6)
    features["eng_commit_message_quality"]= round(commit_q,     6)
    features["eng_pr_discipline"]         = round(pr_disc,      6)
    features["eng_review_participation"]  = review_p

    return features


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_feature_vector(raw_json: Dict[str, Any]) -> Dict[str, float]:
    """Build a complete, flat numeric feature vector from a raw profile dict."""
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

    # Data Quality Audit: check for None, NaN, or non-finite values
    invalid_keys = [k for k, v in vector.items() if v is None or not math.isfinite(v)]
    if len(invalid_keys) > 2:
        logger.warning(
            f"[{username}] Data quality warning: {len(invalid_keys)} feature values are None/NaN/non-finite: {invalid_keys}"
        )
    for k in invalid_keys:
        vector[k] = 0.0

    logger.info(f"[{username}] Total features computed: {len(vector)}")
    return vector


# ---------------------------------------------------------------------------
# Batch Processing Mode
# ---------------------------------------------------------------------------

def _print_batch_summary(
    total_processed: int,
    succeeded_count: int,
    failed_count: int,
    failed_users: Dict[str, str],
    zero_total_repos_users: List[str],
    zero_orig_repos_users: List[str],
    quality_warning_users: List[str],
    stats_df: Optional[Any] = None,
) -> None:
    """Print clean batch execution statistics and summary distributions."""
    sep = "=" * 80
    print(f"\n{sep}")
    print("  DEV LENS - BATCH FEATURE ENGINEERING SUMMARY")
    print(sep)
    print(f" Total Users Processed : {total_processed}")
    print(f" Succeeded             : {succeeded_count}")
    print(f" Failed                : {failed_count}")

    if failed_users:
        print("\n  [FAILED USERS & REASONS]")
        for user, err in failed_users.items():
            print(f"   • {user:<30}: {err}")

    print("\n" + "-" * 80)
    print("  EDGE CASE & DATA QUALITY FLAGS")
    print("-" * 80)
    print(f" Zero Total Repos (Empty Account) : {len(zero_total_repos_users)}")
    if zero_total_repos_users:
        print(f"   Users: {', '.join(zero_total_repos_users)}")

    print(f" Zero Original Repos (Forks Only) : {len(zero_orig_repos_users)}")
    if zero_orig_repos_users:
        print(f"   Users: {', '.join(zero_orig_repos_users)}")

    print(f" Data Quality Warnings (>2 NaNs)  : {len(quality_warning_users)}")
    if quality_warning_users:
        print(f"   Users: {', '.join(quality_warning_users)}")

    if stats_df is not None and not stats_df.empty:
        print("\n" + "-" * 80)
        print("  FEATURE DISTRIBUTION SUMMARY STATISTICS (across all succeeded users)")
        print("-" * 80)
        print(f"{'Feature Name':<38} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 80)
        for fname, row in stats_df.iterrows():
            print(f"{fname:<38} {row['mean']:>10.4f} {row['std']:>10.4f} {row['min']:>10.4f} {row['max']:>10.4f}")

    print(f"\n{sep}\n")


def run_batch_feature_engineering(data_dir: str, save_db: bool = True) -> Dict[str, Any]:
    """Process all {username}.json files in data_dir, compute feature vectors,
    write each result to the features DB table, and print batch summary statistics.
    """
    if not os.path.exists(data_dir):
        print(f"ERROR: Batch directory not found: {data_dir}")
        raise SystemExit(1)

    json_files = [
        f for f in os.listdir(data_dir)
        if f.endswith(".json") and f not in ("batch_summary.json", "features_summary.json")
    ]
    json_files.sort()

    logger.info(f"Starting batch feature engineering for {len(json_files)} files in '{data_dir}'...")

    total_processed = len(json_files)
    succeeded_count = 0
    failed_count = 0
    failed_users: Dict[str, str] = {}

    zero_total_repos_users: List[str] = []
    zero_orig_repos_users: List[str] = []
    quality_warning_users: List[str] = []

    all_vectors: List[Dict[str, float]] = []

    if save_db:
        from devlens.db.session import get_session, init_db, get_engine
        from devlens.db import repository as repo
        from datetime import datetime
        engine = init_db(get_engine())
    else:
        engine = None

    for fname in json_files:
        filepath = os.path.join(data_dir, fname)
        username = os.path.splitext(fname)[0]

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)

            repos = raw.get("repositories", []) or []
            if not isinstance(repos, list):
                repos = []
            original = [r for r in repos if isinstance(r, dict) and not r.get("is_fork")]

            if len(repos) == 0:
                zero_total_repos_users.append(username)
            elif len(original) == 0:
                zero_orig_repos_users.append(username)

            vector = build_feature_vector(raw)

            if save_db:
                with get_session(engine) as session:
                    dev = repo.upsert_developer(
                        session, username=username, resume_source=raw.get("collected_at")
                    )
                    snap = repo.insert_snapshot(
                        session, developer_id=dev.id, raw_json_path=filepath, collected_at=datetime.utcnow()
                    )
                    repo.insert_features(session, snap.id, vector)

                    doc_score = vector.get("doc_score", 0.0)
                    eng_score = vector.get("eng_maturity_score", 0.0)

                    repo.upsert_doc_score(
                        session, snap.id, score=doc_score,
                        components={k: v for k, v in vector.items() if k.startswith("doc_")}
                    )
                    repo.upsert_eng_maturity_score(
                        session, snap.id, score=eng_score,
                        components={k: v for k, v in vector.items() if k.startswith("eng_")}
                    )

            all_vectors.append(vector)
            succeeded_count += 1

        except Exception as e:
            failed_count += 1
            failed_users[username] = str(e)
            logger.error(f"[{username}] Unhandled batch error processing '{fname}': {e}")
            continue

    stats_df = None
    if all_vectors:
        df = pd.DataFrame(all_vectors)
        stats_summary = {
            "mean": df.mean(),
            "std": df.std(),
            "min": df.min(),
            "max": df.max(),
        }
        stats_df = pd.DataFrame(stats_summary)

    _print_batch_summary(
        total_processed=total_processed,
        succeeded_count=succeeded_count,
        failed_count=failed_count,
        failed_users=failed_users,
        zero_total_repos_users=zero_total_repos_users,
        zero_orig_repos_users=zero_orig_repos_users,
        quality_warning_users=quality_warning_users,
        stats_df=stats_df,
    )

    return {
        "total_processed": total_processed,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "failed_users": failed_users,
        "zero_total_repos_users": zero_total_repos_users,
        "zero_orig_repos_users": zero_orig_repos_users,
        "quality_warning_users": quality_warning_users,
        "stats": stats_df,
    }


# ---------------------------------------------------------------------------
# Feature Scaling and Preprocessing for Phase 5 Clustering
# ---------------------------------------------------------------------------

def prepare_scaled_feature_matrix(
    df: pd.DataFrame,
    drop_zero_variance: bool = True
) -> tuple[pd.DataFrame, Any, List[str]]:
    """Prepare scaled feature matrix for ML clustering / Phase 5.
    
    1. Audits and drops zero-variance columns (e.g. lang_primary_is_go, lang_primary_is_rust).
    2. Scales remaining features using StandardScaler.
    3. Verifies no NaN or Inf values remain.
    
    Args:
        df: Unscaled feature DataFrame (developers as index, feature names as columns).
        drop_zero_variance: If True, automatically drop columns where variance == 0.
        
    Returns:
        (scaled_df, scaler, dropped_cols)
    """
    from sklearn.preprocessing import StandardScaler

    if df.empty:
        raise ValueError("Cannot scale an empty feature matrix.")

    dropped_cols = []
    if drop_zero_variance:
        variances = df.var()
        zero_var_cols = list(variances[variances == 0].index)
        if zero_var_cols:
            logger.info(f"Dropping {len(zero_var_cols)} zero-variance columns before scaling: {zero_var_cols}")
            dropped_cols.extend(zero_var_cols)
            df = df.drop(columns=zero_var_cols)

    # Create single repo_has_stars binary flag before dropping raw star features
    if "repo_total_stars" in df.columns:
        df["repo_has_stars"] = (df["repo_total_stars"] > 0).astype(float)
    elif "repo_max_stars" in df.columns:
        df["repo_has_stars"] = (df["repo_max_stars"] > 0).astype(float)

    # Apply explicit exclusions (redundant & raw sparse features)
    exclude_cols = [
        "collab_issues_closed", 
        "collab_issue_close_rate", 
        "collab_review_count", 
        "collab_review_rate_per_pr",
        "doc_score",
        "eng_maturity_score",
        "collab_pr_merge_rate",
        "repo_max_stars",
        "repo_mean_stars",
        "repo_total_stars"
    ]
    cols_to_drop = [c for c in exclude_cols if c in df.columns]
    if cols_to_drop:
        logger.info(f"Excluding redundant features from clustering matrix: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        dropped_cols.extend(cols_to_drop)

    # Binarize highly skewed, near-zero-variance counts
    binarize_cols = ["collab_issues_opened", "collab_issues_commented"]
    for c in binarize_cols:
        if c in df.columns:
            logger.info(f"Binarizing feature: {c}")
            df[c] = (df[c] > 0).astype(float)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)

    nan_count = int(scaled_df.isna().sum().sum())
    inf_count = int((~np.isfinite(scaled_df.values)).sum())

    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"Scaled feature matrix contains invalid values! NaNs: {nan_count}, Infs: {inf_count}")

    logger.info(f"Feature matrix scaled successfully. Shape: {scaled_df.shape} (NaNs: {nan_count}, Infs: {inf_count})")
    return scaled_df, scaler, dropped_cols


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
        description="Compute DevLens feature vector from a raw JSON profile or batch directory."
    )
    parser.add_argument("json_path", nargs="?", default=None, help="Path to raw JSON file (data/raw/<username>.json)")
    parser.add_argument("--batch", type=str, default=None, help="Path to directory containing raw JSON files (e.g. data/raw/)")
    parser.add_argument("--save-db", action="store_true", help="Write features to the database via the repository")
    args = parser.parse_args()

    if args.batch:
        run_batch_feature_engineering(args.batch, save_db=True)
    elif args.json_path:
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

                doc_score = vector.get("doc_score", 0.0)
                eng_score = vector.get("eng_maturity_score", 0.0)

                repo.upsert_doc_score(
                    session, snap.id, score=doc_score,
                    components={k: v for k, v in vector.items() if k.startswith("doc_")}
                )
                repo.upsert_eng_maturity_score(
                    session, snap.id, score=eng_score,
                    components={k: v for k, v in vector.items() if k.startswith("eng_")}
                )

            print(f"  [OK] Saved {count} features to DB for '{username}' (snapshot written)\n")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

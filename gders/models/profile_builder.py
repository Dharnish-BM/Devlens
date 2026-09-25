"""
GDERS Phase 6B Developer Expertise Profile Builder.

Synthesizes fine-grained, auditable developer expertise profiles by aggregating
ground-truth gold comments and confidence-calibrated model predictions across the 10 GDERS categories:
- Transparent evidence scoring formula: Score = Σ (Comment_Type_Weight × Confidence_Weight) × PR_Diversity_Multiplier
- Clear evidence tiers: 'strong_evidence', 'supported_evidence', 'emerging_evidence', 'insufficient_evidence'
- Full comment-level traceability for every category profile
- Preserves exact distinction between gold labels and predicted labels
- Generates data/gders/processed/developer_expertise_profiles.jsonl and expertise_profile_report.json
"""

import collections
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from gders.config import DEFAULT_CONFIG, GDERSConfig
from gders.data.annotation_manager import VALID_TAXONOMY_CATEGORIES
from gders.data.dataset_loader import GDERSDatasetLoader

logger = logging.getLogger(__name__)

# Evidence weights
COMMENT_TYPE_WEIGHTS = {
    "gold": 1.0,
    "high_confidence_pred": 0.70,
    "medium_confidence_pred": 0.40,
    "low_confidence_pred": 0.10,
    "abstained_pred": 0.0,
}


@dataclass
class CategoryExpertiseProfile:
    """Fine-grained category-level expertise profile with full traceability."""
    category_id: str
    evidence_score: float = 0.0
    recommendation_eligible_score: float = 0.0
    evidence_tier: str = "insufficient_evidence"  # insufficient_evidence, emerging_evidence, supported_evidence, strong_evidence
    recommendation_eligible: bool = False
    gold_comment_count: int = 0
    predicted_comment_count: int = 0
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    distinct_pr_count: int = 0
    distinct_repository_count: int = 0
    supporting_comment_ids: List[int] = field(default_factory=list)
    recommendation_eligible_comment_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeveloperExpertiseProfile:
    """Comprehensive Developer Expertise Profile representation."""
    developer_login: str
    total_review_comments: int = 0
    repositories_reviewed: List[str] = field(default_factory=list)
    pull_requests_reviewed: List[int] = field(default_factory=list)
    top_expertise_categories: List[str] = field(default_factory=list)
    category_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    has_sufficient_evidence: bool = False
    identity_class: str = "human_candidate"  # 'human_candidate', 'bot_or_service_account', 'uncertain'
    developer_recommendation_eligible: bool = False
    generated_timestamp: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProfileBuilder:
    """Synthesizes auditable developer expertise profiles from gold & inferred review comments."""

    def __init__(self, config: Optional[GDERSConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        self.loader = GDERSDatasetLoader(config=self.config)

    def load_gold_comments_by_reviewer(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all Phase 5C gold comments mapped by reviewer login."""
        gold_file = self.config.expanded_gold_dataset_file
        gold_by_reviewer: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)

        if gold_file and gold_file.exists():
            with open(gold_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        reviewer = rec.get("commenter_login", "unknown")
                        if reviewer and reviewer != "unknown":
                            gold_by_reviewer[reviewer].append(rec)
        return gold_by_reviewer

    def load_predictions_by_reviewer(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all Phase 6A comment predictions mapped by reviewer login."""
        pred_file = self.config.comment_predictions_file
        pred_by_reviewer: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)

        if pred_file and pred_file.exists():
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        reviewer = rec.get("reviewer_login", "unknown")
                        if reviewer and reviewer != "unknown":
                            pred_by_reviewer[reviewer].append(rec)
        return pred_by_reviewer

    def compute_category_evidence(
        self,
        category: str,
        gold_comments: List[Dict[str, Any]],
        predicted_comments: List[Dict[str, Any]],
        excluded_comment_ids: Optional[Set[int]] = None,
        allowed_evidence_types: Optional[Set[str]] = None,
    ) -> CategoryExpertiseProfile:
        """
        Compute transparent evidence score and assign evidence tier for a specific category.
        Score formula:
            Base_Evidence = Σ (Weight_i)
            PR_Diversity_Factor = min(1.5, 1.0 + 0.1 * (distinct_prs - 1)) if distinct_prs >= 1 else 1.0
            Repo_Diversity_Factor = min(1.2, 1.0 + 0.1 * (distinct_repos - 1)) if distinct_repos >= 1 else 1.0
            Evidence_Score = round(Base_Evidence * PR_Diversity_Factor * Repo_Diversity_Factor, 3)
        """
        excluded = excluded_comment_ids or set()
        allowed_types = allowed_evidence_types or {"gold", "high_confidence_pred", "medium_confidence_pred", "low_confidence_pred"}

        supporting_cids: List[int] = []
        eligible_cids: List[int] = []
        distinct_prs: Set[int] = set()
        distinct_repos: Set[str] = set()
        eligible_prs: Set[int] = set()
        eligible_repos: Set[str] = set()

        gold_count = 0
        high_conf_count = 0
        med_conf_count = 0
        low_conf_count = 0
        base_evidence = 0.0
        eligible_base_evidence = 0.0

        # 1. Process gold comments (Always recommendation-eligible if allowed)
        if "gold" in allowed_types:
            for gc in gold_comments:
                cid = gc.get("comment_id", 0)
                if cid in excluded:
                    continue
                labels = gc.get("category_labels", gc.get("gold_labels", []))
                if category in labels:
                    supporting_cids.append(cid)
                    eligible_cids.append(cid)
                    pr_num = gc.get("pull_request_number")
                    repo_name = gc.get("repository")
                    if pr_num:
                        distinct_prs.add(pr_num)
                        eligible_prs.add(pr_num)
                    if repo_name:
                        distinct_repos.add(repo_name)
                        eligible_repos.add(repo_name)
                    gold_count += 1
                    base_evidence += COMMENT_TYPE_WEIGHTS["gold"]
                    eligible_base_evidence += COMMENT_TYPE_WEIGHTS["gold"]

        # 2. Process predicted comments
        for pc in predicted_comments:
            cid = pc.get("comment_id", 0)
            if cid in excluded:
                continue
            pred_cats = pc.get("predicted_categories", [])
            status = pc.get("prediction_status", "abstained")
            if category in pred_cats and status != "abstained":
                pr_num = pc.get("pull_request_number")
                repo_name = pc.get("repository")

                if status == "high_confidence" and "high_confidence_pred" in allowed_types:
                    supporting_cids.append(cid)
                    if pr_num: distinct_prs.add(pr_num)
                    if repo_name: distinct_repos.add(repo_name)
                    high_conf_count += 1
                    base_evidence += COMMENT_TYPE_WEIGHTS["high_confidence_pred"]
                    eligible_base_evidence += COMMENT_TYPE_WEIGHTS["high_confidence_pred"]
                    eligible_cids.append(cid)
                    if pr_num: eligible_prs.add(pr_num)
                    if repo_name: eligible_repos.add(repo_name)
                elif status == "medium_confidence" and "medium_confidence_pred" in allowed_types:
                    supporting_cids.append(cid)
                    if pr_num: distinct_prs.add(pr_num)
                    if repo_name: distinct_repos.add(repo_name)
                    med_conf_count += 1
                    base_evidence += COMMENT_TYPE_WEIGHTS["medium_confidence_pred"]
                    eligible_base_evidence += COMMENT_TYPE_WEIGHTS["medium_confidence_pred"]
                    eligible_cids.append(cid)
                    if pr_num: eligible_prs.add(pr_num)
                    if repo_name: eligible_repos.add(repo_name)
                elif status == "low_confidence" and "low_confidence_pred" in allowed_types:
                    supporting_cids.append(cid)
                    if pr_num: distinct_prs.add(pr_num)
                    if repo_name: distinct_repos.add(repo_name)
                    low_conf_count += 1
                    base_evidence += COMMENT_TYPE_WEIGHTS["low_confidence_pred"]
                    # Low-confidence is audit-only; excluded from eligible_base_evidence

        total_predicted = high_conf_count + med_conf_count + low_conf_count

        # Apply PR & Repository Diversity scaling
        pr_mult = min(1.5, 1.0 + 0.1 * max(0, len(distinct_prs) - 1)) if distinct_prs else 1.0
        repo_mult = min(1.2, 1.0 + 0.1 * max(0, len(distinct_repos) - 1)) if distinct_repos else 1.0
        final_score = round(base_evidence * pr_mult * repo_mult, 3)

        # Recommendation-eligible score (excludes low-confidence predictions)
        el_pr_mult = min(1.5, 1.0 + 0.1 * max(0, len(eligible_prs) - 1)) if eligible_prs else 1.0
        el_repo_mult = min(1.2, 1.0 + 0.1 * max(0, len(eligible_repos) - 1)) if eligible_repos else 1.0
        final_eligible_score = round(eligible_base_evidence * el_pr_mult * el_repo_mult, 3)

        # Evidence tier determination (based on eligible high-quality evidence)
        # strong_evidence: eligible_score >= 4.0 and >= 2 distinct PRs
        # supported_evidence: eligible_score >= 2.0 and >= 2 high-quality comments
        # emerging_evidence: eligible_score >= 0.70
        # insufficient_evidence: eligible_score < 0.70
        if final_eligible_score >= 4.0 and len(eligible_prs) >= 2:
            tier = "strong_evidence"
            is_eligible = True
        elif final_eligible_score >= 2.0 and (gold_count + high_conf_count + med_conf_count) >= 2:
            tier = "supported_evidence"
            is_eligible = True
        elif final_eligible_score >= 0.70:
            tier = "emerging_evidence"
            is_eligible = False
        else:
            tier = "insufficient_evidence"
            is_eligible = False

        return CategoryExpertiseProfile(
            category_id=category,
            evidence_score=final_score,
            recommendation_eligible_score=final_eligible_score,
            evidence_tier=tier,
            recommendation_eligible=is_eligible,
            gold_comment_count=gold_count,
            predicted_comment_count=total_predicted,
            high_confidence_count=high_conf_count,
            medium_confidence_count=med_conf_count,
            low_confidence_count=low_conf_count,
            distinct_pr_count=len(distinct_prs),
            distinct_repository_count=len(distinct_repos),
            supporting_comment_ids=supporting_cids,
            recommendation_eligible_comment_ids=eligible_cids,
        )

    def build_all_developer_profiles(
        self,
        excluded_comment_ids: Optional[Set[int]] = None,
        allowed_evidence_types: Optional[Set[str]] = None,
        persist: bool = True,
    ) -> Tuple[List[DeveloperExpertiseProfile], Dict[str, Any]]:
        """
        Build expertise profiles for all reviewers in the frozen GDERS corpus.
        Optionally excludes specific comment IDs (for held-out evaluation) and restricts evidence types.
        Saves profiles to data/gders/processed/developer_expertise_profiles.jsonl when persist=True.
        """
        gold_by_reviewer = self.load_gold_comments_by_reviewer()
        pred_by_reviewer = self.load_predictions_by_reviewer()

        all_reviewers = sorted(set(gold_by_reviewer.keys()).union(set(pred_by_reviewer.keys())))
        if persist:
            logger.info(f"Building expertise profiles for {len(all_reviewers)} unique reviewers...")

        profiles: List[DeveloperExpertiseProfile] = []
        categories = sorted(VALID_TAXONOMY_CATEGORIES)

        tier_counts = collections.Counter()
        category_profile_counts = collections.Counter()
        sufficient_evidence_reviewers = 0
        single_comment_reviewers = 0

        for login in all_reviewers:
            g_comments = gold_by_reviewer.get(login, [])
            p_comments = pred_by_reviewer.get(login, [])
            all_user_comments = g_comments + p_comments

            total_comments = len(all_user_comments)
            if total_comments == 1:
                single_comment_reviewers += 1

            repos = sorted(set(c.get("repository", "") for c in all_user_comments if c.get("repository")))
            prs = sorted(set(c.get("pull_request_number", 0) for c in all_user_comments if c.get("pull_request_number")))

            cat_profiles: Dict[str, Dict[str, Any]] = {}
            active_categories: List[Tuple[str, float, str]] = []

            for cat in categories:
                cp = self.compute_category_evidence(
                    category=cat,
                    gold_comments=g_comments,
                    predicted_comments=p_comments,
                    excluded_comment_ids=excluded_comment_ids,
                    allowed_evidence_types=allowed_evidence_types,
                )
                cat_profiles[cat] = cp.to_dict()
                tier_counts[cp.evidence_tier] += 1
                if cp.evidence_tier in ("strong_evidence", "supported_evidence", "emerging_evidence"):
                    category_profile_counts[cat] += 1
                    active_categories.append((cat, cp.evidence_score, cp.evidence_tier))

            # Rank top categories by evidence score descending
            active_categories.sort(key=lambda x: x[1], reverse=True)
            top_cats = [x[0] for x in active_categories[:3]]

            has_sufficient = any(
                cp["evidence_tier"] in ("strong_evidence", "supported_evidence")
                for cp in cat_profiles.values()
            )
            if has_sufficient:
                sufficient_evidence_reviewers += 1

            # Classify identity: 'human_candidate', 'bot_or_service_account', 'uncertain'
            login_lower = login.lower().strip()
            if login_lower.endswith("[bot]") or login_lower in ("copilot", "github-actions", "dependabot", "codecov"):
                identity_cls = "bot_or_service_account"
            elif any(marker in login_lower for marker in ("-bot", "_bot", "service-account", "automation-bot")):
                identity_cls = "uncertain"
            else:
                identity_cls = "human_candidate"

            is_dev_eligible = (identity_cls == "human_candidate") and has_sufficient

            profile = DeveloperExpertiseProfile(
                developer_login=login,
                total_review_comments=total_comments,
                repositories_reviewed=repos,
                pull_requests_reviewed=prs,
                top_expertise_categories=top_cats,
                category_profiles=cat_profiles,
                has_sufficient_evidence=has_sufficient,
                identity_class=identity_cls,
                developer_recommendation_eligible=is_dev_eligible,
            )
            profiles.append(profile)

        # Persist profiles
        prof_file = self.config.developer_expertise_profiles_file
        if persist and prof_file:
            prof_file.parent.mkdir(parents=True, exist_ok=True)
            with open(prof_file, "w", encoding="utf-8") as f:
                for p in profiles:
                    f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(profiles)} developer profiles to {prof_file}")

        # Compile report
        report = {
            "phase": "Phase 6B — Developer Expertise Profiles & Validation",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "profile_summary": {
                "total_unique_reviewers": len(profiles),
                "reviewers_with_sufficient_evidence": sufficient_evidence_reviewers,
                "reviewers_with_insufficient_evidence": len(profiles) - sufficient_evidence_reviewers,
                "reviewers_with_single_comment": single_comment_reviewers,
                "sufficient_evidence_rate": round(sufficient_evidence_reviewers / len(profiles), 4) if profiles else 0.0,
            },
            "evidence_tier_distribution": dict(tier_counts),
            "category_profile_coverage": dict(category_profile_counts),
            "scoring_methodology": {
                "gold_weight": COMMENT_TYPE_WEIGHTS["gold"],
                "high_conf_pred_weight": COMMENT_TYPE_WEIGHTS["high_confidence_pred"],
                "medium_conf_pred_weight": COMMENT_TYPE_WEIGHTS["medium_confidence_pred"],
                "low_conf_pred_weight": COMMENT_TYPE_WEIGHTS["low_confidence_pred"],
                "pr_diversity_multiplier": "1.0 + 0.1 * (distinct_prs - 1) [max 1.5]",
                "repo_diversity_multiplier": "1.0 + 0.1 * (distinct_repos - 1) [max 1.2]",
            },
            "evidence_tiers": {
                "strong_evidence": "Score >= 4.0 and >= 2 distinct PRs",
                "supported_evidence": "Score >= 2.0 and >= 2 comments",
                "emerging_evidence": "Score >= 0.70",
                "insufficient_evidence": "Score < 0.70",
            }
        }

        report_file = self.config.expertise_profile_report_file
        if persist and report_file:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved expertise profile validation report to {report_file}")

        return profiles, report

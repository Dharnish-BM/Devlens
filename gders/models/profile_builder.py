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
    evidence_tier: str = "insufficient_evidence"  # insufficient_evidence, emerging_evidence, supported_evidence, strong_evidence
    gold_comment_count: int = 0
    predicted_comment_count: int = 0
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    distinct_pr_count: int = 0
    distinct_repository_count: int = 0
    supporting_comment_ids: List[int] = field(default_factory=list)

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
    ) -> CategoryExpertiseProfile:
        """
        Compute transparent evidence score and assign evidence tier for a specific category.
        Score formula:
            Base_Evidence = Σ (Weight_i)
            PR_Diversity_Factor = min(1.5, 1.0 + 0.1 * (distinct_prs - 1)) if distinct_prs >= 1 else 1.0
            Repo_Diversity_Factor = min(1.2, 1.0 + 0.1 * (distinct_repos - 1)) if distinct_repos >= 1 else 1.0
            Evidence_Score = round(Base_Evidence * PR_Diversity_Factor * Repo_Diversity_Factor, 3)
        """
        supporting_cids: List[int] = []
        distinct_prs: Set[int] = set()
        distinct_repos: Set[str] = set()

        gold_count = 0
        high_conf_count = 0
        med_conf_count = 0
        low_conf_count = 0
        base_evidence = 0.0

        # 1. Process gold comments
        for gc in gold_comments:
            labels = gc.get("category_labels", gc.get("gold_labels", []))
            if category in labels:
                cid = gc.get("comment_id", 0)
                supporting_cids.append(cid)
                if gc.get("pull_request_number"):
                    distinct_prs.add(gc.get("pull_request_number"))
                if gc.get("repository"):
                    distinct_repos.add(gc.get("repository"))
                gold_count += 1
                base_evidence += COMMENT_TYPE_WEIGHTS["gold"]

        # 2. Process predicted comments
        for pc in predicted_comments:
            pred_cats = pc.get("predicted_categories", [])
            status = pc.get("prediction_status", "abstained")
            if category in pred_cats and status != "abstained":
                cid = pc.get("comment_id", 0)
                supporting_cids.append(cid)
                if pc.get("pull_request_number"):
                    distinct_prs.add(pc.get("pull_request_number"))
                if pc.get("repository"):
                    distinct_repos.add(pc.get("repository"))

                if status == "high_confidence":
                    high_conf_count += 1
                    base_evidence += COMMENT_TYPE_WEIGHTS["high_confidence_pred"]
                elif status == "medium_confidence":
                    med_conf_count += 1
                    base_evidence += COMMENT_TYPE_WEIGHTS["medium_confidence_pred"]
                elif status == "low_confidence":
                    low_conf_count += 1
                    base_evidence += COMMENT_TYPE_WEIGHTS["low_confidence_pred"]

        total_predicted = high_conf_count + med_conf_count + low_conf_count

        # Apply PR & Repository Diversity scaling
        pr_mult = min(1.5, 1.0 + 0.1 * max(0, len(distinct_prs) - 1)) if distinct_prs else 1.0
        repo_mult = min(1.2, 1.0 + 0.1 * max(0, len(distinct_repos) - 1)) if distinct_repos else 1.0
        final_score = round(base_evidence * pr_mult * repo_mult, 3)

        # Evidence tier determination
        # strong_evidence: score >= 4.0 and >= 2 distinct PRs
        # supported_evidence: score >= 2.0 and >= 2 comments
        # emerging_evidence: score >= 0.70 (at least 1 gold or high-conf comment)
        # insufficient_evidence: score < 0.70
        if final_score >= 4.0 and len(distinct_prs) >= 2:
            tier = "strong_evidence"
        elif final_score >= 2.0 and (gold_count + high_conf_count + med_conf_count) >= 2:
            tier = "supported_evidence"
        elif final_score >= 0.70:
            tier = "emerging_evidence"
        else:
            tier = "insufficient_evidence"

        return CategoryExpertiseProfile(
            category_id=category,
            evidence_score=final_score,
            evidence_tier=tier,
            gold_comment_count=gold_count,
            predicted_comment_count=total_predicted,
            high_confidence_count=high_conf_count,
            medium_confidence_count=med_conf_count,
            low_confidence_count=low_conf_count,
            distinct_pr_count=len(distinct_prs),
            distinct_repository_count=len(distinct_repos),
            supporting_comment_ids=supporting_cids,
        )

    def build_all_developer_profiles(self) -> Tuple[List[DeveloperExpertiseProfile], Dict[str, Any]]:
        """
        Build expertise profiles for all reviewers in the frozen GDERS corpus.
        Saves profiles to data/gders/processed/developer_expertise_profiles.jsonl.
        """
        gold_by_reviewer = self.load_gold_comments_by_reviewer()
        pred_by_reviewer = self.load_predictions_by_reviewer()

        all_reviewers = sorted(set(gold_by_reviewer.keys()).union(set(pred_by_reviewer.keys())))
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

            profile = DeveloperExpertiseProfile(
                developer_login=login,
                total_review_comments=total_comments,
                repositories_reviewed=repos,
                pull_requests_reviewed=prs,
                top_expertise_categories=top_cats,
                category_profiles=cat_profiles,
                has_sufficient_evidence=has_sufficient,
            )
            profiles.append(profile)

        # Persist profiles
        prof_file = self.config.developer_expertise_profiles_file
        if prof_file:
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
        if report_file:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved expertise profile validation report to {report_file}")

        return profiles, report

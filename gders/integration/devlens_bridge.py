"""Read-only GDERS to DevLens application-layer bridge.

The bridge uses the frozen GDERS developer profile artifact only. It does not
build profiles, run inference, write artifacts, or access the DevLens database.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from gders.config import DEFAULT_CONFIG, GDERSConfig
from gders.models.profile_builder import DeveloperExpertiseProfile


@dataclass
class GDERSBridgeResponse:
    """Normalized, read-only representation for a DevLens profile surface."""

    github_username: str
    has_gders_evidence: bool
    status: str = "INSUFFICIENT_PR_EXPERTISE_EVIDENCE"
    recommendation_eligible: bool = False
    expertise_profiles: List[Dict[str, Any]] = field(default_factory=list)
    supporting_prs: List[int] = field(default_factory=list)
    supporting_repositories: List[str] = field(default_factory=list)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable copy of the normalized response."""
        return {
            "github_username": self.github_username,
            "has_gders_evidence": self.has_gders_evidence,
            "status": self.status,
            "recommendation_eligible": self.recommendation_eligible,
            "expertise_profiles": [dict(profile) for profile in self.expertise_profiles],
            "supporting_prs": list(self.supporting_prs),
            "supporting_repositories": list(self.supporting_repositories),
            "evidence_summary": dict(self.evidence_summary),
        }


class DevLensGDERSBridge:
    """Expose frozen GDERS profiles through a read-only username lookup."""

    def __init__(self, config: Optional[GDERSConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._profiles_by_username: Optional[Dict[str, DeveloperExpertiseProfile]] = None
        self._evidence_by_comment_id: Optional[Dict[int, Dict[str, Any]]] = None

    @staticmethod
    def _normalize_username(username: str) -> str:
        return (username or "").strip().lstrip("@").lower()

    def _load_profiles(self) -> Dict[str, DeveloperExpertiseProfile]:
        """Load persisted GDERS profiles without rebuilding or persisting them."""
        if self._profiles_by_username is not None:
            return self._profiles_by_username

        profile_file = self.config.developer_expertise_profiles_file
        profiles: Dict[str, DeveloperExpertiseProfile] = {}
        if profile_file and profile_file.exists():
            with open(profile_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    profile = DeveloperExpertiseProfile(**json.loads(line))
                    profiles[self._normalize_username(profile.developer_login)] = profile

        self._profiles_by_username = profiles
        return profiles

    def _load_evidence(self) -> Dict[int, Dict[str, Any]]:
        """Index frozen gold and prediction records for auditable evidence views."""
        if self._evidence_by_comment_id is not None:
            return self._evidence_by_comment_id

        evidence: Dict[int, Dict[str, Any]] = {}
        sources = (
            (self.config.expanded_gold_dataset_file, "gold"),
            (self.config.comment_predictions_file, "inferred"),
        )
        for artifact_path, provenance in sources:
            if not artifact_path or not artifact_path.exists():
                continue
            with open(artifact_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    comment_id = record.get("comment_id")
                    if comment_id is None:
                        continue
                    record["_provenance"] = provenance
                    evidence[int(comment_id)] = record

        self._evidence_by_comment_id = evidence
        return evidence

    @staticmethod
    def _category_view(category_id: str, category: Dict[str, Any]) -> Dict[str, Any]:
        """Keep profile output auditable without exposing comment text."""
        return {
            "category_id": category_id,
            "evidence_score": category.get("evidence_score", 0.0),
            "recommendation_eligible_score": category.get("recommendation_eligible_score", 0.0),
            "evidence_tier": category.get("evidence_tier", "insufficient_evidence"),
            "recommendation_eligible": bool(category.get("recommendation_eligible", False)),
            "gold_comment_count": category.get("gold_comment_count", 0),
            "high_confidence_count": category.get("high_confidence_count", 0),
            "medium_confidence_count": category.get("medium_confidence_count", 0),
            "distinct_pr_count": category.get("distinct_pr_count", 0),
            "distinct_repository_count": category.get("distinct_repository_count", 0),
            "supporting_comment_ids": list(category.get("supporting_comment_ids", [])),
            "recommendation_eligible_comment_ids": list(
                category.get("recommendation_eligible_comment_ids", [])
            ),
        }

    def _supporting_evidence(
        self,
        category_id: str,
        comment_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """Return user-facing evidence metadata without internal model fields."""
        evidence = self._load_evidence()
        records: List[Dict[str, Any]] = []
        for comment_id in comment_ids:
            record = evidence.get(int(comment_id))
            if not record:
                continue
            repository = record.get("repository", "")
            pr_number = record.get("pull_request_number")
            reviewer = record.get("commenter_login", record.get("reviewer_login", ""))
            records.append({
                "repository": repository,
                "pull_request_number": pr_number,
                "pr_url": (
                    f"https://github.com/{repository}/pull/{pr_number}"
                    if repository and pr_number else None
                ),
                "reviewer_username": reviewer,
                "comment_id": int(comment_id),
                "comment_text": record.get("original_body", ""),
                "category": category_id,
                "confidence_status": record.get("prediction_status", "gold"),
                "provenance": record.get("_provenance", "unknown"),
            })
        return records

    def get_profile(self, github_username: str) -> Dict[str, Any]:
        """Return a normalized GDERS view for one GitHub username.

        A missing or blank username returns a no-evidence response. This method
        only reads the persisted GDERS profile artifact and never invokes model
        inference, profile construction, DevLens persistence, or GDERS writes.
        """
        username = (github_username or "").strip().lstrip("@")
        profile = self._load_profiles().get(self._normalize_username(username))
        if profile is None:
            return GDERSBridgeResponse(
                github_username=username,
                has_gders_evidence=False,
                status="INSUFFICIENT_PR_EXPERTISE_EVIDENCE",
                evidence_summary={"status": "not_found"},
            ).to_dict()

        expertise_profiles = [
            {
                **self._category_view(category_id, category),
                "supporting_evidence": self._supporting_evidence(
                    category_id,
                    list(category.get("recommendation_eligible_comment_ids", [])),
                ),
            }
            for category_id, category in sorted(profile.category_profiles.items())
            if category.get("recommendation_eligible")
        ]
        eligible_categories = [
            item["category_id"] for item in expertise_profiles if item["recommendation_eligible"]
        ]

        response = GDERSBridgeResponse(
            github_username=profile.developer_login,
            has_gders_evidence=bool(expertise_profiles),
            status=(
                "AVAILABLE"
                if expertise_profiles
                else "INSUFFICIENT_PR_EXPERTISE_EVIDENCE"
            ),
            recommendation_eligible=bool(profile.developer_recommendation_eligible),
            expertise_profiles=expertise_profiles,
            supporting_prs=sorted(set(profile.pull_requests_reviewed)),
            supporting_repositories=sorted(set(profile.repositories_reviewed)),
            evidence_summary={
                "status": "available",
                "identity_class": profile.identity_class,
                "total_review_comments": profile.total_review_comments,
                "top_expertise_categories": list(profile.top_expertise_categories),
                "eligible_categories": eligible_categories,
                "category_count_with_evidence": len(expertise_profiles),
            },
        )
        return response.to_dict()

    def recommend(self, expertise_query: Any, top_k: int = 5) -> Dict[str, Any]:
        """Return separate GDERS recommendations using the existing human gate."""
        from gders.models.recommender import GDERSRecommender

        recommender = GDERSRecommender(config=self.config, profiles=list(self._load_profiles().values()))
        response = recommender.recommend(expertise_query, top_k=top_k, require_human=True)
        return {
            "requested_expertise": response.query,
            "matched_categories": list(response.matched_categories),
            "status": response.status,
            "candidates": [
                {
                    "username": candidate.username,
                    "recommendation_score": candidate.recommendation_score,
                    "matched_categories": list(candidate.matched_categories),
                    "evidence_tier": candidate.evidence_tier,
                    "supporting_evidence_count": len(candidate.supporting_comment_ids),
                    "supporting_comment_ids": list(candidate.supporting_comment_ids),
                    "supporting_prs": candidate.distinct_prs,
                    "supporting_repositories": candidate.distinct_repositories,
                    "explanation": candidate.explanation,
                }
                for candidate in response.candidates
                if candidate.identity_class == "human_candidate"
                and candidate.developer_recommendation_eligible
            ],
        }


_default_bridge: Optional[DevLensGDERSBridge] = None


def get_gders_profile(github_username: str) -> Dict[str, Any]:
    """Convenience lookup using the default frozen GDERS configuration."""
    global _default_bridge
    if _default_bridge is None:
        _default_bridge = DevLensGDERSBridge()
    return _default_bridge.get_profile(github_username)

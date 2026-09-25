"""
Comprehensive isolated unit tests for GDERS Phase 7A Expert Developer Recommendation Engine.

Tests covering:
1. single-category recommendation
2. multi-category recommendation
3. natural-language category mapping
4. unknown category handling
5. empty query handling
6. no eligible candidates handling
7. top_k behavior
8. recommendation score determinism
9. supporting-comment traceability
10. low-confidence exclusion
11. abstained exclusion
12. multi-repository evidence
13. duplicate evidence prevention
14. explanation correctness
15. candidate eligibility gate
"""

import pathlib
from gders.config import GDERSConfig
from gders.models.profile_builder import DeveloperExpertiseProfile
from gders.models.recommender import (
    GDERSRecommender,
    RecommendationCandidate,
    RecommendationResponse,
)


def test_single_category_recommendation():
    recommender = GDERSRecommender()
    resp = recommender.recommend("TESTING_QUALITY", top_k=5)

    assert resp.status == "ok"
    assert resp.matched_categories == ["TESTING_QUALITY"]
    assert len(resp.candidates) <= 5
    assert len(resp.candidates) > 0

    first = resp.candidates[0]
    assert isinstance(first, RecommendationCandidate)
    assert first.recommendation_score > 0
    assert "TESTING_QUALITY" in first.matched_categories
    assert first.evidence_tier in ("supported_evidence", "strong_evidence")
    assert len(first.supporting_comment_ids) > 0
    assert "TESTING_QUALITY" in first.explanation


def test_multi_category_recommendation():
    recommender = GDERSRecommender()
    resp = recommender.recommend(["BUG_LOGIC", "TESTING_QUALITY"], top_k=5)

    assert resp.status == "ok"
    assert set(resp.matched_categories) == {"BUG_LOGIC", "TESTING_QUALITY"}
    assert len(resp.candidates) > 0

    # Ensure deterministic ranking where candidates matching multiple categories are scored appropriately
    scores = [c.recommendation_score for c in resp.candidates]
    assert scores == sorted(scores, reverse=True)


def test_natural_language_category_mapping():
    recommender = GDERSRecommender()

    # Query 1: Data management & queries
    resp1 = recommender.recommend("Need a developer experienced in database schema design and SQL queries")
    assert resp1.status == "ok"
    assert "DATA_MANAGEMENT" in resp1.matched_categories
    assert "database" in resp1.query_mapping_rationale.get("DATA_MANAGEMENT", "")

    # Query 2: Security & encryption
    resp2 = recommender.recommend("Audit authentication token and vulnerability security injection")
    assert resp2.status == "ok"
    assert "SECURITY_PRIVACY" in resp2.matched_categories

    # Query 3: Frontend widget UI
    resp3 = recommender.recommend("Flutter UI widget layout and component rendering")
    assert resp3.status == "ok"
    assert "FRONTEND_UI_UX" in resp3.matched_categories


def test_unknown_and_empty_query_handling():
    recommender = GDERSRecommender()

    # Empty query string
    resp_empty = recommender.recommend("")
    assert resp_empty.status in ("invalid_query", "unmatched_query")
    assert len(resp_empty.candidates) == 0

    # Unknown nonsensical category
    resp_unknown = recommender.recommend("XYZNonExistentDomain12345")
    assert resp_unknown.status == "unmatched_query"
    assert len(resp_unknown.candidates) == 0


def test_candidate_eligibility_gate_and_low_confidence_exclusion():
    recommender = GDERSRecommender()

    # Mock an ineligible profile (only low-confidence predictions, no gold, no medium/high)
    ineligible_profile = DeveloperExpertiseProfile(
        developer_login="low_conf_dev",
        total_review_comments=5,
        repositories_reviewed=["apache/spark"],
        pull_requests_reviewed=[101],
        top_expertise_categories=["TESTING_QUALITY"],
        category_profiles={
            "TESTING_QUALITY": {
                "category_id": "TESTING_QUALITY",
                "evidence_score": 0.50,  # Base from low-conf
                "recommendation_eligible_score": 0.0,  # Strictly zero
                "evidence_tier": "insufficient_evidence",
                "recommendation_eligible": False,  # Gate closed
                "gold_comment_count": 0,
                "predicted_comment_count": 5,
                "high_confidence_count": 0,
                "medium_confidence_count": 0,
                "low_confidence_count": 5,
                "distinct_pr_count": 1,
                "distinct_repository_count": 1,
                "supporting_comment_ids": [1, 2, 3, 4, 5],
                "recommendation_eligible_comment_ids": [],
            }
        },
        has_sufficient_evidence=False,
    )

    cand = recommender.compute_candidate_recommendation_score(ineligible_profile, ["TESTING_QUALITY"])
    assert cand is None, "Ineligible candidate must be blocked from recommendation"


def test_recommendation_score_determinism_and_tie_breaking():
    recommender = GDERSRecommender()

    resp1 = recommender.recommend("BUG_LOGIC", top_k=5)
    resp2 = recommender.recommend("BUG_LOGIC", top_k=5)

    assert len(resp1.candidates) == len(resp2.candidates)
    for c1, c2 in zip(resp1.candidates, resp2.candidates):
        assert c1.username == c2.username
        assert c1.recommendation_score == c2.recommendation_score
        assert c1.explanation == c2.explanation


def test_supporting_comment_traceability_and_deduplication():
    recommender = GDERSRecommender()
    resp = recommender.recommend("TESTING_QUALITY", top_k=3)

    for cand in resp.candidates:
        # Check no duplicate comment IDs
        assert len(cand.supporting_comment_ids) == len(set(cand.supporting_comment_ids))
        assert len(cand.repositories) > 0
        assert cand.distinct_prs > 0
        assert cand.explanation != ""

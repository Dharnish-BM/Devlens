"""
Unit and integration tests for GDERS Phase 7A.5 Candidate Identity & Recommendation Validation Gate.

Tests covering:
1. bot account exclusion
2. [bot] login handling
3. uncertain identity handling
4. human candidate eligibility
5. bot evidence retained in audit data
6. recommendation pool excludes bots
7. identity eligibility separate from expertise eligibility
8. multi-category natural-language mapping
9. ambiguous query handling
10. deterministic recommendation ordering
"""

import json
from gders.config import DEFAULT_CONFIG
from gders.evaluation.identity_auditor import IdentityAuditor
from gders.models.profile_builder import CategoryExpertiseProfile, DeveloperExpertiseProfile, ProfileBuilder
from gders.models.recommender import GDERSRecommender, RecommendationCandidate


def test_identity_classification_bot_and_human():
    auditor = IdentityAuditor()
    
    # 1. Explicit [bot] login
    c1 = auditor.classify_reviewer_identity("gemini-code-assist[bot]", [])
    assert c1["identity_class"] == "bot_or_service_account"
    assert c1["recommendation_candidate_eligible"] is False

    # 2. Known service/AI login
    c2 = auditor.classify_reviewer_identity("Copilot", [])
    assert c2["identity_class"] == "bot_or_service_account"
    assert c2["recommendation_candidate_eligible"] is False

    # 3. Human login
    c3 = auditor.classify_reviewer_identity("astefan", [])
    assert c3["identity_class"] == "human_candidate"
    assert c3["recommendation_candidate_eligible"] is True

    # 4. Uncertain login
    c4 = auditor.classify_reviewer_identity("test-automation-bot-account", [])
    assert c4["identity_class"] == "uncertain"
    assert c4["recommendation_candidate_eligible"] is False


def test_full_corpus_identity_audit_statistics():
    auditor = IdentityAuditor()
    audit_data = auditor.run_full_identity_audit()

    assert audit_data["total_reviewer_identities"] == 151
    assert audit_data["identity_class_breakdown"]["bot_or_service_account"] == 2
    assert audit_data["identity_class_breakdown"]["human_candidate"] == 149
    assert audit_data["identity_class_breakdown"]["uncertain"] == 0
    assert audit_data["recommendation_identity_eligible_count"] == 149

    bot_logins = [r["github_login"] for r in audit_data["reviewers"] if r["identity_class"] == "bot_or_service_account"]
    assert "Copilot" in bot_logins
    assert "gemini-code-assist[bot]" in bot_logins


def test_bot_evidence_retention_in_profiles():
    """Verify that bots have their profiles and evidence preserved, but are marked ineligible for dev recommendation."""
    builder = ProfileBuilder()
    profiles, _ = builder.build_all_developer_profiles()

    copilot_prof = next((p for p in profiles if p.developer_login == "Copilot"), None)
    gemini_prof = next((p for p in profiles if p.developer_login == "gemini-code-assist[bot]"), None)

    assert copilot_prof is not None
    assert copilot_prof.identity_class == "bot_or_service_account"
    assert copilot_prof.has_sufficient_evidence is True
    assert copilot_prof.developer_recommendation_eligible is False
    assert copilot_prof.total_review_comments > 0

    assert gemini_prof is not None
    assert gemini_prof.identity_class == "bot_or_service_account"
    assert gemini_prof.has_sufficient_evidence is True
    assert gemini_prof.developer_recommendation_eligible is False
    assert gemini_prof.total_review_comments > 0


def test_recommendation_pool_excludes_bots():
    """Verify that GDERSRecommender does not recommend bots in human developer mode."""
    recommender = GDERSRecommender()

    # Query where Copilot / gemini had strong scores in Phase 7A
    resp = recommender.recommend(["BUG_LOGIC", "TESTING_QUALITY"], top_k=20)
    assert resp.status == "ok"
    assert len(resp.candidates) > 0

    candidate_logins = [c.username for c in resp.candidates]
    assert "Copilot" not in candidate_logins
    assert "gemini-code-assist[bot]" not in candidate_logins

    for c in resp.candidates:
        assert c.identity_class == "human_candidate"
        assert c.developer_recommendation_eligible is True


def test_identity_eligibility_separate_from_expertise_eligibility():
    """Verify separation of identity eligibility vs expertise evidence eligibility."""
    recommender = GDERSRecommender()

    # Bot profile with strong evidence
    bot_profile = DeveloperExpertiseProfile(
        developer_login="copilot-test-account[bot]",
        total_review_comments=10,
        repositories_reviewed=["test/repo"],
        pull_requests_reviewed=[1, 2],
        category_profiles={
            "TESTING_QUALITY": {
                "recommendation_eligible": True,
                "recommendation_eligible_score": 10.0,
                "evidence_tier": "strong_evidence",
                "gold_comment_count": 5,
                "high_confidence_count": 5,
                "medium_confidence_count": 0,
                "supporting_comment_ids": [101, 102],
            }
        },
        has_sufficient_evidence=True,
        identity_class="bot_or_service_account",
        developer_recommendation_eligible=False,
    )

    # Human profile with insufficient evidence
    human_ineligible_profile = DeveloperExpertiseProfile(
        developer_login="human-no-evidence",
        total_review_comments=1,
        repositories_reviewed=["test/repo"],
        pull_requests_reviewed=[1],
        category_profiles={
            "TESTING_QUALITY": {
                "recommendation_eligible": False,
                "recommendation_eligible_score": 0.4,
                "evidence_tier": "insufficient_evidence",
                "gold_comment_count": 0,
                "high_confidence_count": 0,
                "medium_confidence_count": 1,
                "supporting_comment_ids": [103],
            }
        },
        has_sufficient_evidence=False,
        identity_class="human_candidate",
        developer_recommendation_eligible=False,
    )

    # Standard human recommendation excludes both
    c_bot = recommender.compute_candidate_recommendation_score(bot_profile, ["TESTING_QUALITY"], require_human=True)
    assert c_bot is None

    c_human_inel = recommender.compute_candidate_recommendation_score(human_ineligible_profile, ["TESTING_QUALITY"], require_human=True)
    assert c_human_inel is None

    # Audit mode (require_human=False) allows bot scoring for transparency
    c_bot_audit = recommender.compute_candidate_recommendation_score(bot_profile, ["TESTING_QUALITY"], require_human=False)
    assert c_bot_audit is not None
    assert c_bot_audit.identity_class == "bot_or_service_account"
    assert c_bot_audit.developer_recommendation_eligible is False


def test_natural_language_multi_category_query_mappings():
    """Verify deterministic query mapping for required requirement test cases."""
    recommender = GDERSRecommender()

    cases = [
        ("database schema", ["DATA_MANAGEMENT"]),
        ("SQL optimization", ["DATA_MANAGEMENT", "PERF_OPTIMIZATION"]),
        ("backend testing", ["TESTING_QUALITY"]),
        ("security authentication", ["SECURITY_PRIVACY"]),
        ("frontend React", ["FRONTEND_UI_UX"]),
        ("architecture design", ["ARCH_DESIGN"]),
        ("Need a developer skilled in database schema design and SQL optimization", ["ARCH_DESIGN", "DATA_MANAGEMENT", "PERF_OPTIMIZATION"]),
        ("UnmatchedRequirementXyzFooBar", []),
    ]

    for q, expected in cases:
        cats, rats, status = recommender.map_query_to_categories(q)
        assert set(cats) == set(expected), f"Query '{q}' failed: got {cats}, expected {expected}"
        if expected:
            assert status == "ok"
            for c in cats:
                assert c in rats
        else:
            assert len(cats) == 0


def test_deterministic_recommendation_ordering():
    """Verify recommendation ranking determinism across repeated calls."""
    recommender = GDERSRecommender()

    resp1 = recommender.recommend(["BUG_LOGIC", "TESTING_QUALITY"], top_k=5)
    resp2 = recommender.recommend(["BUG_LOGIC", "TESTING_QUALITY"], top_k=5)

    assert resp1.candidates == resp2.candidates
    assert [c.username for c in resp1.candidates] == [c.username for c in resp2.candidates]
    assert [c.recommendation_score for c in resp1.candidates] == [c.recommendation_score for c in resp2.candidates]


def test_validation_gate_runner_execution():
    """Verify the validation gate reports current data and its legacy verdict."""
    auditor = IdentityAuditor()
    report, verdict = auditor.run_validation_gate()

    assert verdict == "PHASE 7B BLOCKED — Validation checks failed"
    assert report["reviewer_identity_summary"]["total_reviewer_identities"] == 151
    assert report["reviewer_identity_summary"]["bot_or_service_accounts"] == 2
    assert report["candidate_pool_audit"]["recommendation_candidates_after_identity_filtering"] == 68
    assert report["query_mapping_audit"]["all_mappings_passed"] is True
    assert DEFAULT_CONFIG.reviewer_identity_audit_file.exists()
    assert DEFAULT_CONFIG.phase7a5_validation_report_file.exists()
    assert DEFAULT_CONFIG.phase7a5_validation_report_md_file.exists()

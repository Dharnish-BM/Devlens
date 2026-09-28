import json

from gders.config import GDERSConfig
from gders.integration.devlens_bridge import DevLensGDERSBridge
from gders.models.profile_builder import DeveloperExpertiseProfile


def make_config(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    profile = DeveloperExpertiseProfile(
        developer_login="ExampleUser",
        total_review_comments=3,
        repositories_reviewed=["owner/repo"],
        pull_requests_reviewed=[12, 14],
        top_expertise_categories=["TESTING_QUALITY"],
        category_profiles={
            "TESTING_QUALITY": {
                "evidence_score": 2.4,
                "recommendation_eligible_score": 2.4,
                "evidence_tier": "supported_evidence",
                "recommendation_eligible": True,
                "gold_comment_count": 2,
                "high_confidence_count": 1,
                "medium_confidence_count": 0,
                "distinct_pr_count": 2,
                "distinct_repository_count": 1,
                "supporting_comment_ids": [101, 102, 103],
                "recommendation_eligible_comment_ids": [101, 102, 103],
            }
        },
        has_sufficient_evidence=True,
        identity_class="human_candidate",
        developer_recommendation_eligible=True,
    )
    with open(config.developer_expertise_profiles_file, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(profile.to_dict()) + "\n")
    with open(config.expanded_gold_dataset_file, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "comment_id": 101,
            "repository": "owner/repo",
            "pull_request_number": 12,
            "commenter_login": "ExampleUser",
            "original_body": "Please add a regression test.",
            "category_labels": ["TESTING_QUALITY"],
        }) + "\n")
    return config


def test_bridge_returns_normalized_read_only_profile(tmp_path):
    bridge = DevLensGDERSBridge(make_config(tmp_path))

    result = bridge.get_profile("@exampleuser")

    assert result["github_username"] == "ExampleUser"
    assert result["has_gders_evidence"] is True
    assert result["status"] == "AVAILABLE"
    assert result["recommendation_eligible"] is True
    assert result["supporting_prs"] == [12, 14]
    assert result["supporting_repositories"] == ["owner/repo"]
    assert result["expertise_profiles"][0]["category_id"] == "TESTING_QUALITY"
    assert result["expertise_profiles"][0]["supporting_comment_ids"] == [101, 102, 103]
    evidence = result["expertise_profiles"][0]["supporting_evidence"][0]
    assert evidence["repository"] == "owner/repo"
    assert evidence["pr_url"] == "https://github.com/owner/repo/pull/12"
    assert evidence["comment_text"] == "Please add a regression test."
    assert evidence["provenance"] == "gold"


def test_bridge_returns_not_found_without_building_profiles(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    bridge = DevLensGDERSBridge(config)

    result = bridge.get_profile("missing-user")

    assert result == {
        "github_username": "missing-user",
        "has_gders_evidence": False,
        "status": "INSUFFICIENT_PR_EXPERTISE_EVIDENCE",
        "recommendation_eligible": False,
        "expertise_profiles": [],
        "supporting_prs": [],
        "supporting_repositories": [],
        "evidence_summary": {"status": "not_found"},
    }
    assert not config.developer_expertise_profiles_file.exists()


def test_bridge_does_not_expose_comment_text_or_write_artifacts(tmp_path):
    config = make_config(tmp_path)
    profile_path = config.developer_expertise_profiles_file
    before = profile_path.read_bytes()
    bridge = DevLensGDERSBridge(config)

    result = bridge.get_profile("ExampleUser")

    assert "original_body" not in json.dumps(result)
    assert profile_path.read_bytes() == before


def test_bridge_recommendation_keeps_human_identity_gate(tmp_path):
    config = make_config(tmp_path)
    bridge = DevLensGDERSBridge(config)

    result = bridge.recommend("TESTING_QUALITY", top_k=5)

    assert result["matched_categories"] == ["TESTING_QUALITY"]
    assert [candidate["username"] for candidate in result["candidates"]] == ["ExampleUser"]
    assert all(candidate["username"] != "Copilot" for candidate in result["candidates"])


def test_bridge_marks_profile_without_qualifying_categories_insufficient(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    profile = DeveloperExpertiseProfile(
        developer_login="SparseUser",
        total_review_comments=1,
        category_profiles={
            "BUG_LOGIC": {
                "supporting_comment_ids": [501],
                "recommendation_eligible": False,
                "evidence_tier": "emerging_evidence",
            }
        },
        identity_class="human_candidate",
        developer_recommendation_eligible=False,
    )
    with open(config.developer_expertise_profiles_file, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(profile.to_dict()) + "\n")

    result = DevLensGDERSBridge(config).get_profile("SparseUser")

    assert result["has_gders_evidence"] is False
    assert result["status"] == "INSUFFICIENT_PR_EXPERTISE_EVIDENCE"
    assert result["expertise_profiles"] == []

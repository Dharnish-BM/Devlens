"""
Unit tests for GDERS Repository Suitability Assessment.
All tests use mock fixtures and do NOT execute live GitHub API calls.
"""

from pathlib import Path
from unittest.mock import MagicMock
from gders.config import GDERSConfig, TargetRepository
from gders.data.repository_assessment import (
    RepositoryAssessmentRunner,
    RepoPilotMetrics,
    is_technical_comment_heuristic,
)


def test_technical_comment_heuristic():
    # Technical review comments
    tech1 = "Please refactor this API method to avoid race conditions in the async handler."
    is_t1, reason1 = is_technical_comment_heuristic(tech1)
    assert is_t1 is True
    assert "technical_signal" in reason1

    tech2 = "We should add a unit test with mock fixture to ensure `get_user()` returns None on 404."
    is_t2, _ = is_technical_comment_heuristic(tech2)
    assert is_t2 is True

    # Conversational / Noise comments
    noise1 = "LGTM!"
    is_n1, reason_n1 = is_technical_comment_heuristic(noise1)
    assert is_n1 is False
    assert "too_short" in reason_n1 or "noise" in reason_n1

    noise2 = "looks good to me"
    is_n2, reason_n2 = is_technical_comment_heuristic(noise2)
    assert is_n2 is False
    assert "noise" in reason_n2

    noise3 = "👍"
    is_n3, _ = is_technical_comment_heuristic(noise3)
    assert is_n3 is False


def test_evaluate_single_repository_mocked(tmp_path):
    config = GDERSConfig(
        base_data_dir=tmp_path / "gders",
        pilot_max_prs_per_repo=5,
    )
    mock_client = MagicMock()

    # 1. Metadata response
    mock_client.rest_request.side_effect = [
        # Metadata
        {
            "stargazers_count": 80000,
            "forks_count": 32000,
            "open_issues_count": 120,
            "language": "Python",
            "description": "The Web framework for perfectionists with deadlines.",
            "pushed_at": "2026-09-20T10:00:00Z",
        },
        # PR list (2 PRs)
        [
            {
                "number": 1,
                "title": "Fix SQL injection in QuerySet extra()",
                "user": {"login": "contributor1", "id": 10},
                "created_at": "2026-06-01T00:00:00Z",
                "updated_at": "2026-06-02T00:00:00Z",
                "closed_at": None,
                "merged_at": None,
                "state": "open",
                "html_url": "https://github.com/django/django/pull/1",
            },
            {
                "number": 2,
                "title": "Minor typo in docs",
                "user": {"login": "contributor2", "id": 20},
                "created_at": "2026-06-03T00:00:00Z",
                "updated_at": "2026-06-04T00:00:00Z",
                "closed_at": None,
                "merged_at": None,
                "state": "open",
                "html_url": "https://github.com/django/django/pull/2",
            },
        ],
        # PR #1 Comments (2 comments)
        [
            {
                "id": 1001,
                "user": {"login": "reviewer1", "id": 101},
                "body": "Ensure the query parameters are properly sanitized before binding to the SQL statement.",
                "created_at": "2026-06-01T12:00:00Z",
                "updated_at": "2026-06-01T12:00:00Z",
                "html_url": "https://github.com/django/django/pull/1#c1001",
                "pull_request_url": "https://api.github.com/repos/django/django/pulls/1",
                "commit_id": "sha1",
                "path": "django/db/models/query.py",
                "diff_hunk": "@@ -1,3 +1,3 @@",
            },
            {
                "id": 1002,
                "user": {"login": "reviewer2", "id": 102},
                "body": "Can you also add an assertion in `test_query.py`?",
                "created_at": "2026-06-01T13:00:00Z",
                "updated_at": "2026-06-01T13:00:00Z",
                "html_url": "https://github.com/django/django/pull/1#c1002",
                "pull_request_url": "https://api.github.com/repos/django/django/pulls/1",
                "commit_id": "sha1",
                "path": "django/db/models/query.py",
                "diff_hunk": "@@ -1,3 +1,3 @@",
            },
        ],
        # PR #2 Comments (0 comments)
        [],
    ]

    runner = RepositoryAssessmentRunner(config=config, client=mock_client)
    target = TargetRepository(owner="django", name="django")
    metrics, comments = runner.evaluate_single_repository(target, max_prs=2)

    assert metrics.repository_name == "django/django"
    assert metrics.primary_language == "Python"
    assert metrics.stars == 80000
    assert metrics.sampled_prs == 2
    assert metrics.sampled_prs_with_review_comments == 1
    assert metrics.sampled_review_comments == 2
    assert metrics.sampled_unique_reviewers == 2
    assert metrics.sampled_technical_comments == 2
    assert metrics.technical_signal_ratio == 1.0
    assert len(comments) == 2


def test_run_assessment_report_generation(tmp_path):
    config = GDERSConfig(
        base_data_dir=tmp_path / "gders",
        target_repositories=[TargetRepository(owner="django", name="django")],
    )
    mock_client = MagicMock()
    mock_client.rest_request.side_effect = [
        # Metadata
        {"stargazers_count": 80000, "forks_count": 32000, "language": "Python", "description": "Django web framework"},
        # PR list
        [],
    ]

    runner = RepositoryAssessmentRunner(config=config, client=mock_client)
    report = runner.run_assessment(max_prs_per_repo=1)

    assert "aggregate_statistics" in report
    assert "repositories" in report
    assert len(report["repositories"]) == 1
    assert (config.pilot_processed_dir / "repository_suitability.json").exists()
    assert (config.pilot_processed_dir / "repository_suitability.md").exists()


def test_replacement_repository_assessment_mocked(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    mock_client = MagicMock()
    mock_client.rest_request.side_effect = [
        # scikit-learn metadata
        {"stargazers_count": 60000, "language": "Python", "description": "Machine learning in Python"},
        # PR list
        [{"number": 10, "title": "Fix PCA solver", "created_at": "2026-05-01T00:00:00Z", "user": {"login": "dev1"}}],
        # Comments
        [{"id": 200, "user": {"login": "rev1"}, "body": "Please add typing annotation for the solver argument.", "created_at": "2026-05-01T01:00:00Z"}],
    ]

    runner = RepositoryAssessmentRunner(config=config, client=mock_client)
    replacements = [TargetRepository(owner="scikit-learn", name="scikit-learn")]
    report = runner.run_assessment(
        repositories=replacements,
        max_prs_per_repo=1,
        output_prefix="replacement_repository_suitability",
        report_title="GDERS Replacement Repository Suitability Assessment",
    )

    assert len(report["repositories"]) == 1
    assert report["repositories"][0]["repository_name"] == "scikit-learn/scikit-learn"
    assert (config.pilot_processed_dir / "replacement_repository_suitability.json").exists()
    assert (config.pilot_processed_dir / "replacement_repository_suitability.md").exists()


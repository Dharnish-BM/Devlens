"""
Unit tests for GDERS repository collection abstractions.
All tests use mocks and do NOT make live GitHub API requests.
"""

import json
from unittest.mock import MagicMock
from gders.config import GDERSConfig, TargetRepository
from gders.data.repo_collector import GDERSRepoCollector, ReviewCommentRecord


def test_gders_config_repository_count():
    config = GDERSConfig()
    assert len(config.target_repositories) == 10
    repo_names = [r.full_name for r in config.target_repositories]
    assert "apache/spark" in repo_names
    assert "elastic/elasticsearch" in repo_names
    assert "scikit-learn/scikit-learn" in repo_names
    assert "facebook/react" in repo_names


def test_gders_repo_collector_init(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    mock_client = MagicMock()
    collector = GDERSRepoCollector(config=config, client=mock_client)

    assert collector.config == config
    assert collector.client == mock_client
    assert config.raw_comments_dir.exists()


def test_iter_pull_requests_pagination(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders", per_page_pagination=1)
    mock_client = MagicMock()
    # Page 1 returns 1 PR, Page 2 returns 1 PR, Page 3 returns empty
    mock_client.rest_request.side_effect = [
        [
            {
                "number": 101,
                "title": "PR 101",
                "user": {"login": "alice", "id": 1},
                "created_at": "2026-05-01T12:00:00Z",
                "updated_at": "2026-05-02T12:00:00Z",
                "state": "closed",
                "html_url": "https://github.com/rails/rails/pull/101",
            }
        ],
        [
            {
                "number": 102,
                "title": "PR 102",
                "user": {"login": "bob", "id": 2},
                "created_at": "2026-05-03T12:00:00Z",
                "updated_at": "2026-05-04T12:00:00Z",
                "state": "closed",
                "html_url": "https://github.com/rails/rails/pull/102",
            }
        ],
        [],
    ]

    collector = GDERSRepoCollector(config=config, client=mock_client)
    prs = list(collector.iter_pull_requests("rails", "rails", max_prs=2))

    assert len(prs) == 2
    assert prs[0].number == 101
    assert prs[1].number == 102
    assert mock_client.rest_request.call_count == 2


def test_fetch_pr_review_comments_provenance(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    mock_client = MagicMock()
    mock_client.rest_request.return_value = [
        {
            "id": 555,
            "user": {"login": "bob", "id": 2},
            "body": "Consider using set instead of list for O(1) lookups.",
            "created_at": "2026-05-02T10:00:00Z",
            "updated_at": "2026-05-02T10:00:00Z",
            "html_url": "https://github.com/rails/rails/pull/101#discussion_r555",
            "pull_request_url": "https://api.github.com/repos/rails/rails/pulls/101",
            "commit_id": "abcdef123456",
            "path": "actionpack/lib/action_dispatch.rb",
            "diff_hunk": "@@ -10,4 +10,4 @@",
        }
    ]

    collector = GDERSRepoCollector(config=config, client=mock_client)
    comments = collector.fetch_pr_review_comments(
        "rails", "rails", pull_number=101,
        pull_request_title="Optimization PR",
        pull_request_author="alice"
    )

    assert len(comments) == 1
    c = comments[0]
    assert isinstance(c, ReviewCommentRecord)
    assert c.repository == "rails/rails"
    assert c.comment_id == 555
    assert c.commenter_login == "bob"
    assert c.file_path == "actionpack/lib/action_dispatch.rb"
    assert c.pull_request_title == "Optimization PR"
    assert c.pull_request_author == "alice"
    assert "set instead of list" in c.comment_body


def test_collection_deduplication_and_resumability(tmp_path):
    config = GDERSConfig(
        base_data_dir=tmp_path / "gders",
        max_prs_per_repo=5,
    )
    mock_client = MagicMock()
    mock_client.api_calls_count = 5

    # Returns 1 PR
    mock_client.rest_request.side_effect = [
        [
            {
                "number": 201,
                "title": "Fix crash",
                "user": {"login": "charlie"},
                "created_at": "2026-05-01T12:00:00Z",
                "updated_at": "2026-05-02T12:00:00Z",
                "state": "open",
                "html_url": "https://github.com/facebook/react/pull/201",
            }
        ],
        # PR 201 comments (2 comments, one duplicate ID in second batch)
        [
            {
                "id": 1001,
                "user": {"login": "dan"},
                "body": "Check null pointer",
                "created_at": "2026-05-01T13:00:00Z",
                "updated_at": "2026-05-01T13:00:00Z",
                "html_url": "https://github.com/facebook/react/pull/201#discussion_r1001",
            },
            {
                "id": 1001,  # Duplicate
                "user": {"login": "dan"},
                "body": "Check null pointer",
                "created_at": "2026-05-01T13:00:00Z",
                "updated_at": "2026-05-01T13:00:00Z",
                "html_url": "https://github.com/facebook/react/pull/201#discussion_r1001",
            },
            {
                "id": 1002,
                "user": {"login": "sophie"},
                "body": "Add regression test",
                "created_at": "2026-05-01T14:00:00Z",
                "updated_at": "2026-05-01T14:00:00Z",
                "html_url": "https://github.com/facebook/react/pull/201#discussion_r1002",
            }
        ]
    ]

    collector = GDERSRepoCollector(config=config, client=mock_client)
    res = collector.collect_repository("facebook", "react")

    assert res["status"] == "completed"
    assert res["review_comments_collected"] == 2  # Deduplicated from 3 to 2
    assert res["unique_reviewers"] == 2

    # Verify JSONL content on disk
    raw_file = config.raw_comments_dir / "facebook__react.jsonl"
    assert raw_file.exists()
    lines = raw_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2

    # Test resumability: running again when completed skips fetching
    mock_client.rest_request.reset_mock()
    res2 = collector.collect_repository("facebook", "react", resume=True)
    assert res2["status"] == "completed"
    mock_client.rest_request.assert_not_called()


def test_dataset_manifest_generation(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    mock_client = MagicMock()
    collector = GDERSRepoCollector(config=config, client=mock_client)

    # Mock state
    state = {
        "repositories": {
            "apache/spark": {
                "repository": "apache/spark",
                "status": "completed",
                "prs_collected": 10,
                "review_comments_collected": 50,
                "unique_reviewers": 8,
                "api_requests": 15,
                "api_errors": 0,
                "rate_limit_events": 0,
                "file_path": str(config.raw_comments_dir / "apache__spark.jsonl"),
            }
        }
    }
    collector.save_collection_state(state)

    # Create dummy jsonl file
    dummy_file = config.raw_comments_dir / "apache__spark.jsonl"
    dummy_file.write_text(
        json.dumps({
            "repository": "apache/spark",
            "comment_id": 999,
            "commenter_login": "matei",
            "pull_request_number": 50,
            "comment_body": "Optimize RDD partition count",
        }) + "\n",
        encoding="utf-8"
    )

    manifest = collector.generate_dataset_manifest()

    assert manifest["dataset_name"] == "GDERS-10-Corpus"
    assert manifest["summary_statistics"]["total_repositories"] == 1
    assert manifest["summary_statistics"]["total_prs_collected"] == 10
    assert manifest["summary_statistics"]["total_review_comments_collected"] == 50
    assert manifest["status_summary"]["completed"] == ["apache/spark"]
    assert config.dataset_manifest_file.exists()


def test_api_failure_and_malformed_response_handling(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    mock_client = MagicMock()
    # 1. PR listing returns None or malformed non-list
    mock_client.rest_request.side_effect = [
        {"error": "Internal Server Error"},  # not a list
    ]

    collector = GDERSRepoCollector(config=config, client=mock_client)
    res = collector.collect_repository("elastic", "elasticsearch")

    assert res["status"] == "completed"
    assert res["prs_collected"] == 0
    assert res["review_comments_collected"] == 0


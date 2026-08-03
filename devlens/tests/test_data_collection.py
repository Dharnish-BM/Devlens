import os
import json
import pytest
from unittest.mock import MagicMock, patch
from devlens.data_collection.github_client import GitHubClient
from devlens.data_collection.collect_profile import GitHubProfileCollector


def test_github_client_init():
    client = GitHubClient(token="fake_token")
    assert client.token == "fake_token"
    assert "Authorization" in client.session.headers
    assert client.session.headers["Authorization"] == "Bearer fake_token"


@patch("requests.Session.get")
def test_rest_request_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"login": "torvalds"}
    mock_response.headers = {}
    mock_get.return_value = mock_response

    client = GitHubClient(token="fake_token")
    res = client.rest_request("users/torvalds")

    assert res == {"login": "torvalds"}
    assert client.rest_calls_count == 1
    assert client.api_calls_count == 1


@patch("requests.Session.post")
def test_graphql_request_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "user": {
                "login": "torvalds"
            }
        }
    }
    mock_response.headers = {}
    mock_post.return_value = mock_response

    client = GitHubClient(token="fake_token")
    res = client.graphql_request("query { user(login: \"torvalds\") { login } }")

    assert res == {"user": {"login": "torvalds"}}
    assert client.graphql_calls_count == 1
    assert client.api_calls_count == 1


@patch("devlens.data_collection.collect_profile.GitHubClient")
def test_collect_full_profile(mock_client_cls, tmp_path):
    mock_client = mock_client_cls.return_value
    mock_client.rest_request.side_effect = [
        {"login": "testuser", "public_repos": 1},  # user_info
        [{"id": 1, "name": "testrepo", "language": "Python", "stargazers_count": 10, "forks_count": 2, "size": 100, "created_at": "2025-01-01T00:00:00Z", "pushed_at": "2025-01-02T00:00:00Z", "fork": False, "default_branch": "main"}],  # repos
        [{"total": 50, "week": 1600000000}],  # commit_activity
        {"total_count": 5},  # pr_opened
        {"total_count": 4},  # pr_merged
        {"total_count": 2},  # pr_reviewed
        {"total_count": 1},  # issue_opened
        {"total_count": 1},  # issue_closed
        {"total_count": 3},  # issue_commented
    ]
    mock_client.graphql_request.return_value = {
        "user": {
            "login": "testuser",
            "contributionsCollection": {
                "contributionCalendar": {"totalContributions": 120}
            },
            "pinnedItems": {"nodes": []},
            "repositories": {"nodes": []}
        }
    }
    mock_client.get_api_calls_summary.return_value = {
        "total_calls": 9,
        "rest_calls": 8,
        "graphql_calls": 1
    }

    collector = GitHubProfileCollector(client=mock_client)
    profile = collector.collect_full_profile("testuser")

    assert profile["username"] == "testuser"
    assert profile["summary_metrics"]["repo_count"] == 1
    assert profile["summary_metrics"]["total_commits"] == 120

    saved_file = collector.save_profile_data(profile, output_dir=str(tmp_path))
    assert os.path.exists(saved_file)
    with open(saved_file, "r") as f:
        data = json.load(f)
        assert data["username"] == "testuser"

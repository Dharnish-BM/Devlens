import os
import pytest
from unittest.mock import MagicMock, patch
from devlens.data_collection.resume_extractor import ResumeExtractor


def test_find_direct_url_username():
    extractor = ResumeExtractor()
    text = "Check out my work at https://github.com/torvalds or git."
    username = extractor._find_direct_url_username(text)
    assert username == "torvalds"


def test_find_labeled_username():
    extractor = ResumeExtractor()
    text = "Linus Torvalds\nGitHub Profile: torvalds\nLocation: OR"
    username = extractor._find_labeled_username(text)
    assert username == "torvalds"


@patch("devlens.data_collection.resume_extractor.GitHubClient")
def test_infer_and_verify_username_success(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.rest_request.side_effect = lambda endpoint: {"login": "torvalds"} if "torvalds" in endpoint else None

    extractor = ResumeExtractor(client=mock_client)
    text = "Linus Torvalds\nEmail: torvalds@kernel.org"
    username = extractor._infer_and_verify_username(text)

    assert username == "torvalds"


@patch("devlens.data_collection.resume_extractor.GitHubClient")
def test_extract_github_username_none(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.rest_request.return_value = None

    extractor = ResumeExtractor(client=mock_client)

    # Use existing sample resume or test dict
    res = extractor.extract_github_username("data/sample_resumes/resume_no_match.docx")
    assert res["username"] is None
    assert res["confidence"] == "none"

"""
Unit tests for GDERS NLP preprocessing and dataset loading.
"""

from pathlib import Path
from gders.config import GDERSConfig
from gders.data.dataset_loader import GDERSDatasetLoader
from gders.data.nlp_preprocessor import GDERSNLPPreprocessor, ProcessedCommentRecord
from gders.data.repo_collector import ReviewCommentRecord


def test_url_removal():
    preprocessor = GDERSNLPPreprocessor()
    raw = "Refer to https://example.com/api/v1 or www.docs.io for implementation details."
    cleaned = preprocessor.clean_text(raw)
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert "www.docs.io" not in cleaned
    assert "refer" in cleaned
    assert "implementation" in cleaned


def test_lowercasing_and_whitespace_normalization():
    preprocessor = GDERSNLPPreprocessor()
    raw = "   UPPERCASE   Word    With   MULTIPLE    Spaces!  \n \t "
    cleaned = preprocessor.clean_text(raw)
    assert cleaned == "uppercase word with multiple spaces"


def test_code_block_and_inline_code_handling():
    preprocessor = GDERSNLPPreprocessor()
    raw = "Review this `calculate_hash()` helper: ```def foo(): return 42``` Please test it."
    cleaned = preprocessor.clean_text(raw)
    # Inline code content is retained as natural tokens, fenced code block is stripped
    assert "calculate_hash" in cleaned
    assert "return 42" not in cleaned
    assert "def foo" not in cleaned
    assert "please test" in cleaned


def test_mention_and_issue_normalization():
    preprocessor = GDERSNLPPreprocessor()
    raw = "Hey @octocat please review this fix for #4562 and verify with @teammate."
    cleaned = preprocessor.clean_text(raw)
    assert "octocat" not in cleaned
    assert "@" not in cleaned
    assert "#4562" not in cleaned
    assert "4562" not in cleaned
    assert "please review this fix" in cleaned


def test_tokenization_and_stopword_removal():
    preprocessor = GDERSNLPPreprocessor()
    raw = "This is a comprehensive test of our query execution performance in the database."
    cleaned = preprocessor.clean_text(raw)
    tokens = preprocessor.tokenize(cleaned)

    assert "comprehensive" in tokens
    assert "test" in tokens
    assert "query" in tokens
    assert "execution" in tokens
    assert "performance" in tokens
    assert "database" in tokens
    # Stopwords like 'this', 'is', 'a', 'of', 'our', 'in', 'the' should be excluded
    for stop in ["this", "is", "a", "of", "our", "in", "the"]:
        assert stop not in tokens


def test_deterministic_lemmatization():
    preprocessor = GDERSNLPPreprocessor()
    assert preprocessor.lemmatize("queries") in ("query", "queries")
    assert preprocessor.lemmatize("running") in ("run", "running")
    assert preprocessor.lemmatize("optimized") in ("optimize", "optimized")


def test_technical_token_preservation():
    preprocessor = GDERSNLPPreprocessor()
    tech_words = [
        "api", "http", "json", "sql", "kubernetes", "docker",
        "grpc", "async", "await", "oauth", "cache", "thread",
        "mutex", "database", "query", "schema", "endpoint"
    ]
    raw = " ".join(tech_words)
    cleaned = preprocessor.clean_text(raw)
    tokens = preprocessor.tokenize(cleaned)
    lemmas = [preprocessor.lemmatize(t) for t in tokens]

    for tw in tech_words:
        assert any(tw in lem or lem in tw for lem in lemmas), f"Technical term '{tw}' was destroyed!"


def test_empty_and_low_information_comment_detection():
    preprocessor = GDERSNLPPreprocessor()
    # 1. Pure URL
    rec1 = {"comment_body": "https://github.com/rust-lang/rust"}
    proc1 = preprocessor.process_record(rec1)
    assert proc1.is_empty_after_preprocessing is True
    assert proc1.preprocessing_status == "empty_after_preprocessing"
    assert proc1.token_count == 0

    # 2. Pure punctuation / emoji
    rec2 = {"comment_body": "👍 !!! ??? ..."}
    proc2 = preprocessor.process_record(rec2)
    assert proc2.is_empty_after_preprocessing is True
    assert proc2.token_count == 0

    # 3. Pure mentions
    rec3 = {"comment_body": "@reviewer1 @reviewer2"}
    proc3 = preprocessor.process_record(rec3)
    assert proc3.is_empty_after_preprocessing is True

    # 4. Valid comment
    rec4 = {"comment_body": "Please add unit test for cache invalidation."}
    proc4 = preprocessor.process_record(rec4)
    assert proc4.is_empty_after_preprocessing is False
    assert proc4.preprocessing_status == "success"
    assert proc4.token_count > 0


def test_metadata_preservation_and_original_body_immutability():
    preprocessor = GDERSNLPPreprocessor()
    raw_record = {
        "repository": "flutter/flutter",
        "owner": "flutter",
        "repo": "flutter",
        "repository_url": "https://github.com/flutter/flutter",
        "pull_request_number": 999,
        "pull_request_url": "https://github.com/flutter/flutter/pull/999",
        "pull_request_title": "Fix widget rendering lag",
        "pull_request_author": "dev1",
        "comment_id": 888777,
        "commenter_login": "senior_reviewer",
        "commenter_id": 4321,
        "comment_url": "https://github.com/flutter/flutter/pull/999#r888777",
        "created_at": "2026-03-01T10:00:00Z",
        "updated_at": "2026-03-01T10:05:00Z",
        "commit_id": "c0ffee123",
        "file_path": "packages/flutter/lib/src/rendering/box.dart",
        "diff_hunk": "@@ -10,4 +10,4 @@",
        "collection_timestamp": "2026-09-24T08:26:00Z",
        "collection_window_start": "2025-09-24T08:26:00Z",
        "collection_window_end": "2026-09-24T08:26:00Z",
        "comment_body": "Exact unmodified raw text with `code` & https://flutter.dev URL!"
    }

    proc = preprocessor.process_record(raw_record)

    # Immutability assertion
    assert proc.original_body == raw_record["comment_body"]
    assert proc.comment_id == 888777
    assert proc.commenter_login == "senior_reviewer"
    assert proc.repository == "flutter/flutter"
    assert proc.pull_request_number == 999
    assert proc.file_path == "packages/flutter/lib/src/rendering/box.dart"
    assert proc.commit_id == "c0ffee123"
    assert proc.has_inline_code is True
    assert proc.has_url is True
    assert proc.cleaned_text != proc.original_body


def test_deterministic_reproducibility():
    preprocessor = GDERSNLPPreprocessor()
    text = "Verify async await deadlock handling in mutex locks when thread-safe pool executes."
    res1 = preprocessor.process_record({"comment_body": text})
    res2 = preprocessor.process_record({"comment_body": text})

    assert res1.cleaned_text == res2.cleaned_text
    assert res1.tokens == res2.tokens
    assert res1.lemmas == res2.lemmas
    assert res1.token_count == res2.token_count


def test_malformed_input_handling():
    preprocessor = GDERSNLPPreprocessor()
    # None comment_body
    rec1 = {"comment_body": None}
    proc1 = preprocessor.process_record(rec1)
    assert proc1.original_body == ""
    assert proc1.is_empty_after_preprocessing is True

    # Empty dict
    rec2 = {}
    proc2 = preprocessor.process_record(rec2)
    assert proc2.original_body == ""
    assert proc2.is_empty_after_preprocessing is True


def test_gders_dataset_loader_processed_roundtrip(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    loader = GDERSDatasetLoader(config=config)

    proc_record = ProcessedCommentRecord(
        repository="facebook/react",
        owner="facebook",
        repo="react",
        repository_url="https://github.com/facebook/react",
        pull_request_number=123,
        pull_request_url="https://github.com/facebook/react/pull/123",
        pull_request_title="Fix hooks ordering",
        pull_request_author="alice",
        comment_id=5050,
        commenter_login="dan",
        commenter_id=900,
        comment_url="https://github.com/facebook/react/pull/123#r5050",
        created_at="2026-04-01T12:00:00Z",
        updated_at="2026-04-01T12:00:00Z",
        original_body="Consider using `useMemo` for expensive computations.",
        cleaned_text="consider using usememo for expensive computations",
        processed_text="consider use usememo expensive computation",
        tokens=["consider", "usememo", "expensive", "computations"],
        lemmas=["consider", "usememo", "expensive", "computation"],
        token_count=4,
        unique_token_count=4,
        character_count_original=51,
        character_count_processed=42,
    )

    saved_path = loader.save_processed_comments("facebook", "react", [proc_record])
    assert saved_path.exists()

    loaded = loader.load_processed_comments_for_repo("facebook", "react")
    assert len(loaded) == 1
    assert loaded[0]["comment_id"] == 5050
    assert loaded[0]["original_body"] == "Consider using `useMemo` for expensive computations."
    assert loaded[0]["lemmas"] == ["consider", "usememo", "expensive", "computation"]


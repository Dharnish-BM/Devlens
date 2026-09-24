"""
Unit tests for GDERS Phase 5A Taxonomy Validation & Gold Dataset Annotation.
"""

import json
from pathlib import Path
from gders.config import GDERSConfig
from gders.data.annotation_manager import (
    GDERSAnnotationManager,
    GoldAnnotationRecord,
    VALID_TAXONOMY_CATEGORIES,
    VALID_ANNOTATION_STATUSES,
)


def test_taxonomy_schema_and_integrity():
    config = GDERSConfig()
    tax_file = config.taxonomy_file
    assert tax_file.exists()

    with open(tax_file, "r", encoding="utf-8") as f:
        tax = json.load(f)

    assert "taxonomy_name" in tax
    assert "categories" in tax
    assert len(tax["categories"]) == 10

    category_ids = {c["id"] for c in tax["categories"]}
    assert category_ids == VALID_TAXONOMY_CATEGORIES
    for cat in tax["categories"]:
        assert "name" in cat
        assert "definition" in cat
        assert "inclusion_criteria" in cat
        assert "exclusion_criteria" in cat
        assert "examples" in cat
        assert len(cat["examples"]) >= 2


def test_annotation_guidelines_existence():
    config = GDERSConfig()
    g_file = config.annotation_guidelines_file
    assert g_file.exists()
    content = g_file.read_text(encoding="utf-8")
    assert "GDERS Code Review Comment Annotation Guidelines" in content
    assert "Decision Tree" in content
    assert "ARCH_DESIGN" in content
    assert "TESTING_QUALITY" in content


def test_deterministic_stratified_sampling(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    manager = GDERSAnnotationManager(config=config)

    # Mock processed comments
    mock_comments = [
        {
            "repository": f"repo{i%3}",
            "comment_id": i,
            "original_body": f"Comment body {i}",
            "processed_text": f"comment body {i}",
            "token_count": (i % 30) + 1,
            "has_code_block": bool(i % 2 == 0),
            "has_inline_code": False,
        }
        for i in range(1, 101)
    ]
    manager.loader.save_processed_comments("repo0", "repo0", mock_comments[:33])
    manager.loader.save_processed_comments("repo1", "repo1", mock_comments[33:66])
    manager.loader.save_processed_comments("repo2", "repo2", mock_comments[66:])

    sample1 = manager.create_stratified_sample(target_sample_size=30, random_seed=42)
    sample2 = manager.create_stratified_sample(target_sample_size=30, random_seed=42)

    # Check reproducibility
    assert len(sample1) == len(sample2)
    assert [s["comment_id"] for s in sample1] == [s["comment_id"] for s in sample2]
    # Check no duplicates in sample
    assert len(sample1) == len(set(s["comment_id"] for s in sample1))


def test_gold_annotation_curation_rules():
    manager = GDERSAnnotationManager()

    sample_inputs = [
        # 1. Non-technical
        {"comment_id": 1, "original_body": "LGTM! Thanks 👍", "lemmas": ["lgtm", "thanks"], "is_empty_after_preprocessing": False},
        # 2. Testing QA
        {"comment_id": 2, "original_body": "Please add a unit test for this mock assertion.", "lemmas": ["please", "add", "unit", "test", "mock", "assertion"]},
        # 3. Perf & Multi-label
        {"comment_id": 3, "original_body": "Optimize this database query lookup using index.", "lemmas": ["optimize", "database", "query", "lookup", "use", "index"]},
        # 4. Insufficient context
        {"comment_id": 4, "original_body": "What about this?", "lemmas": ["what"]},
    ]

    annotated = manager.apply_gold_curation(sample_inputs)
    assert len(annotated) == 4

    # Record 1: non_technical
    assert annotated[0].annotation_status == "non_technical"

    # Record 2: labeled (TESTING_QUALITY)
    assert annotated[1].annotation_status == "labeled"
    assert "TESTING_QUALITY" in annotated[1].category_labels

    # Record 3: labeled (Multi-label PERF_OPTIMIZATION & DATA_MANAGEMENT)
    assert annotated[2].annotation_status == "labeled"
    assert "PERF_OPTIMIZATION" in annotated[2].category_labels
    assert "DATA_MANAGEMENT" in annotated[2].category_labels

    # Record 4: insufficient_context
    assert annotated[3].annotation_status == "insufficient_context"


def test_annotation_report_generation_schema(tmp_path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    manager = GDERSAnnotationManager(config=config)

    records = [
        GoldAnnotationRecord(
            comment_id=101,
            repository="flutter/flutter",
            owner="flutter",
            repo="flutter",
            pull_request_number=50,
            commenter_login="alice",
            original_body="Add unit test",
            processed_text="add unit test",
            category_labels=["TESTING_QUALITY"],
            primary_category="TESTING_QUALITY",
            annotation_status="labeled",
        ),
        GoldAnnotationRecord(
            comment_id=102,
            repository="flutter/flutter",
            owner="flutter",
            repo="flutter",
            pull_request_number=50,
            commenter_login="bob",
            original_body="Thanks!",
            processed_text="thanks",
            category_labels=[],
            annotation_status="non_technical",
        )
    ]

    report = manager.generate_annotation_report(records, random_seed=42)

    assert report["phase"] == "Phase 5A - Taxonomy Validation & Gold Dataset"
    assert report["sample_summary"]["total_sampled_comments"] == 2
    assert report["sample_summary"]["labeled_count"] == 1
    assert report["sample_summary"]["non_technical_count"] == 1
    assert "TESTING_QUALITY" in report["category_distribution"]
    assert report["category_distribution"]["TESTING_QUALITY"]["label_count"] == 1

"""
Unit tests for GDERS Phase 6 Full-Corpus Inference and Developer Expertise Profiles.
"""

import json
import pathlib
from gders.config import GDERSConfig
from gders.models.inference_engine import GDERSInferenceEngine, CommentPredictionRecord
from gders.models.profile_builder import ProfileBuilder, DeveloperExpertiseProfile


def test_inference_engine_execution_and_schema():
    engine = GDERSInferenceEngine()
    predictions, summary = engine.run_full_corpus_inference()

    assert len(predictions) == 884
    assert summary["total_unlabeled_inferred"] == 884
    assert "confidence_distribution" in summary
    assert "high_confidence" in summary["confidence_distribution"]
    assert "abstained" in summary["confidence_distribution"]

    first_pred = predictions[0]
    assert isinstance(first_pred, CommentPredictionRecord)
    assert first_pred.comment_id > 0
    assert first_pred.prediction_status in ("high_confidence", "medium_confidence", "low_confidence", "abstained")
    assert isinstance(first_pred.predicted_categories, list)
    assert isinstance(first_pred.prediction_scores, dict)
    assert first_pred.prediction_source == "predicted"


def test_gold_comments_excluded_from_prediction_set():
    engine = GDERSInferenceEngine()
    _, _, gold_ids = engine.load_training_dataset()
    predictions, _ = engine.run_full_corpus_inference()

    pred_ids = {p.comment_id for p in predictions}

    # Verify complete disjointness between gold comments and inferred predictions
    intersection = gold_ids.intersection(pred_ids)
    assert len(intersection) == 0, f"Found overlapping comment IDs: {intersection}"


def test_developer_expertise_profile_builder():
    builder = ProfileBuilder()
    profiles, report = builder.build_all_developer_profiles()

    assert len(profiles) == 151
    assert report["profile_summary"]["total_unique_reviewers"] == 151
    assert report["profile_summary"]["reviewers_with_sufficient_evidence"] > 0

    first_prof = profiles[0]
    assert isinstance(first_prof, DeveloperExpertiseProfile)
    assert first_prof.developer_login != ""
    assert first_prof.total_review_comments > 0
    assert len(first_prof.category_profiles) == 10

    for cat_id, cat_data in first_prof.category_profiles.items():
        assert "evidence_score" in cat_data
        assert "evidence_tier" in cat_data
        assert "supporting_comment_ids" in cat_data
        assert cat_data["evidence_tier"] in ("insufficient_evidence", "emerging_evidence", "supported_evidence", "strong_evidence")


def test_evidence_scoring_and_diversity_scaling():
    builder = ProfileBuilder()

    # Case 1: Gold comment only
    gold_comments = [
        {
            "comment_id": 1001,
            "repository": "apache/spark",
            "pull_request_number": 501,
            "category_labels": ["TESTING_QUALITY"],
        }
    ]
    pred_comments = []

    res = builder.compute_category_evidence("TESTING_QUALITY", gold_comments, pred_comments)
    assert res.gold_comment_count == 1
    assert res.predicted_comment_count == 0
    assert res.evidence_score == 1.0
    assert res.evidence_tier == "emerging_evidence"
    assert res.supporting_comment_ids == [1001]

    # Case 2: Multi-PR diversity boost
    gold_multi = [
        {"comment_id": 1001, "repository": "apache/spark", "pull_request_number": 501, "category_labels": ["TESTING_QUALITY"]},
        {"comment_id": 1002, "repository": "apache/spark", "pull_request_number": 502, "category_labels": ["TESTING_QUALITY"]},
        {"comment_id": 1003, "repository": "apache/spark", "pull_request_number": 503, "category_labels": ["TESTING_QUALITY"]},
        {"comment_id": 1004, "repository": "apache/spark", "pull_request_number": 504, "category_labels": ["TESTING_QUALITY"]},
    ]
    res_multi = builder.compute_category_evidence("TESTING_QUALITY", gold_multi, [])
    assert res_multi.gold_comment_count == 4
    # 4 gold comments = base 4.0, distinct PRs = 4 -> PR multiplier = 1.0 + 0.3 = 1.3 -> 4.0 * 1.3 = 5.2
    assert res_multi.evidence_score == 5.2
    assert res_multi.evidence_tier == "strong_evidence"

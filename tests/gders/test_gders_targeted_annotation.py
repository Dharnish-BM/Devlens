"""
Unit tests for GDERS Phase 5C Targeted Annotation Expansion & Classifier Stabilization.
"""

import json
import pathlib
from gders.config import GDERSConfig
from gders.data.targeted_annotator import (
    GDERSTargetedAnnotator,
    TargetedAnnotationRecord,
    TARGET_CATEGORIES,
)
from gders.evaluation.experiment_runner import GDERSExperimentRunner


def test_targeted_candidate_selection_and_no_duplicates():
    annotator = GDERSTargetedAnnotator()

    # 1. Sample targeted candidates against real corpus
    candidates = annotator.sample_targeted_candidates(
        phase5a_ids=set(),
        target_per_category=5,
        random_seed=42,
    )

    assert len(candidates) > 0
    assert len(candidates) <= 20  # 4 target categories * 5

    # Verify no duplicate comment IDs among candidates
    candidate_ids = [c["comment_id"] for c in candidates]
    assert len(candidate_ids) == len(set(candidate_ids))


def test_curate_targeted_annotations_conforms_to_guidelines():
    annotator = GDERSTargetedAnnotator()
    candidates = [
        {
            "comment_id": 9001,
            "repository": "apache/spark",
            "owner": "apache",
            "repo": "spark",
            "pull_request_number": 101,
            "commenter_login": "dev1",
            "original_body": "We need to decouple the executor lifecycle interface from the storage hierarchy.",
            "processed_text": "need decouple executor lifecycle interface storage hierarchy",
            "lemmas": ["need", "decouple", "executor", "lifecycle", "interface", "storage", "hierarchy"],
            "tokens": ["need", "decouple", "executor", "lifecycle", "interface", "storage", "hierarchy"],
            "candidate_reason": "Matched ARCH_DESIGN cues: decouple, interface",
        },
        {
            "comment_id": 9002,
            "repository": "elastic/elasticsearch",
            "owner": "elastic",
            "repo": "elasticsearch",
            "pull_request_number": 202,
            "commenter_login": "dev2",
            "original_body": "Sanitize this input token to prevent auth token credential leak.",
            "processed_text": "sanitize input token prevent auth token credential leak",
            "lemmas": ["sanitize", "input", "token", "prevent", "auth", "token", "credential", "leak"],
            "tokens": ["sanitize", "input", "token", "prevent", "auth", "token", "credential", "leak"],
            "candidate_reason": "Matched SECURITY_PRIVACY cues: auth, token",
        },
        {
            "comment_id": 9003,
            "repository": "rails/rails",
            "owner": "rails",
            "repo": "rails",
            "pull_request_number": 303,
            "commenter_login": "dev3",
            "original_body": "LGTM! Thanks!",
            "processed_text": "lgtm thanks",
            "lemmas": ["lgtm", "thanks"],
            "tokens": ["lgtm", "thanks"],
            "candidate_reason": "Inspect noise",
        }
    ]

    records = annotator.curate_targeted_annotations(candidates)
    assert len(records) == 3

    # Record 1: ARCH_DESIGN
    assert records[0].annotation_status == "labeled"
    assert "ARCH_DESIGN" in records[0].gold_labels
    assert records[0].provenance == "phase_5c"

    # Record 2: SECURITY_PRIVACY
    assert records[1].annotation_status == "labeled"
    assert "SECURITY_PRIVACY" in records[1].gold_labels

    # Record 3: Non-technical
    assert records[2].annotation_status == "non_technical"
    assert len(records[2].gold_labels) == 0


def test_expanded_dataset_provenance_and_schema(tmp_path: pathlib.Path):
    config = GDERSConfig(base_data_dir=tmp_path / "gders")
    config.ensure_directories()
    annotator = GDERSTargetedAnnotator(config=config)

    # Mock Phase 5A sample
    p5a_sample = [
        {
            "comment_id": 1,
            "repository": "facebook/react",
            "original_body": "Fix null render pointer in state hook",
            "processed_text": "fix null render pointer state hook",
            "category_labels": ["BUG_LOGIC", "FRONTEND_UI_UX"],
            "annotation_status": "labeled",
        }
    ]
    with open(config.annotation_sample_file, "w", encoding="utf-8") as f:
        for r in p5a_sample:
            f.write(json.dumps(r) + "\n")

    # Mock Phase 5C records
    p5c_record = TargetedAnnotationRecord(
        comment_id=2,
        repository="kubernetes/kubernetes",
        owner="kubernetes",
        repo="kubernetes",
        pull_request_number=55,
        commenter_login="k8s_dev",
        original_body="Add database migration index on status column",
        processed_text="add database migration index status column",
        gold_labels=["DATA_MANAGEMENT"],
        annotation_status="labeled",
        provenance="phase_5c",
    )

    expanded_records, meta = annotator.build_expanded_dataset([p5c_record])

    assert len(expanded_records) == 2
    assert meta["total_expanded_labeled_comments"] == 2
    assert meta["phase_5a_labeled_count"] == 1
    assert meta["phase_5c_labeled_count"] == 1
    assert expanded_records[0]["provenance"] == "phase_5a"
    assert expanded_records[1]["provenance"] == "phase_5c"


def test_phase5c_comparative_experiment_runner():
    runner = GDERSExperimentRunner()

    # Run comparative experiment on live Phase 5 artifacts
    comp = runner.run_phase5c_comparative_experiment()

    assert "dataset_comparison" in comp
    assert comp["dataset_comparison"]["phase_5b_labeled_comments"] == 185
    assert comp["dataset_comparison"]["phase_5c_expanded_labeled_comments"] == 341
    assert "low_support_category_analysis" in comp
    assert "ARCH_DESIGN" in comp["low_support_category_analysis"]
    assert "SECURITY_PRIVACY" in comp["low_support_category_analysis"]
    assert "DATA_MANAGEMENT" in comp["low_support_category_analysis"]
    assert "PERF_OPTIMIZATION" in comp["low_support_category_analysis"]

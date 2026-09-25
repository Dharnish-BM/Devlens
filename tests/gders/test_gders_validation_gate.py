"""
Unit tests for GDERS Phase 6.5 Validation Gate and Methodological Auditing.
"""

from gders.evaluation.validation_gate import Phase65ValidationGate


def test_validation_gate_leakage_and_boundary_checks():
    gate = Phase65ValidationGate()
    leakage = gate.verify_data_leakage_boundaries()

    assert leakage["total_corpus_comments"] == 1225
    assert leakage["gold_comments_count"] == 341
    assert leakage["inferred_comments_count"] == 884
    assert leakage["overlap_count"] == 0
    assert leakage["is_disjoint"] is True
    assert leakage["is_exact_corpus_cover"] is True
    assert leakage["status"] == "PASSED"


def test_validation_gate_comment_traceability():
    gate = Phase65ValidationGate()
    trace = gate.verify_comment_traceability()

    assert trace["total_profile_comment_references"] > 1000
    assert trace["missing_comment_references"] == 0
    assert trace["reviewer_identity_mismatches"] == 0
    assert trace["status"] == "PASSED"


def test_validation_gate_evidence_sensitivity_analysis():
    gate = Phase65ValidationGate()
    sens = gate.run_evidence_sensitivity_analysis()

    assert "scheme_a_full_profile_view" in sens
    assert "scheme_b_recommendation_eligible" in sens
    assert "scheme_c_strict_gold_high_only" in sens

    # Verify that scheme A has >= scheme B >= scheme C eligible reviewers
    u_a = sens["scheme_a_full_profile_view"]["eligible_reviewers"]
    u_b = sens["scheme_b_recommendation_eligible"]["eligible_reviewers"]
    u_c = sens["scheme_c_strict_gold_high_only"]["eligible_reviewers"]

    assert u_a >= u_b >= u_c
    assert u_a == 71
    assert u_b == 69
    assert u_c == 61


def test_validation_gate_full_report_clearance():
    gate = Phase65ValidationGate()
    report = gate.generate_phase6_5_report()

    assert report["gate_decision"] == "PHASE 7 CLEARED"
    assert report["status"] == "PHASE 7 CLEARED"

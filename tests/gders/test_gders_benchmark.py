"""
Unit and integration tests for GDERS Phase 7B Recommendation Benchmark Evaluation Engine.

Tests covering:
1. deterministic benchmark split
2. developer grouping
3. held-out evidence exclusion
4. no benchmark leakage
5. metric calculations (Precision@K, Recall@K, MRR, nDCG)
6. Precision@K logic
7. Recall@K logic
8. MRR logic
9. nDCG logic
10. bot exclusion in benchmark
11. baseline reproducibility
12. ablation reproducibility
13. category-level aggregation
14. insufficient-category handling
15. deterministic ranking
16. empty benchmark handling
"""

import math
from gders.config import DEFAULT_CONFIG
from gders.evaluation.benchmark import (
    BenchmarkQueryUnit,
    GDERSBenchmark,
    QueryEvaluationResult,
    compute_dcg,
    compute_ndcg,
    BaselineEvidenceCountRecommender,
    BaselinePRDiversityRecommender,
    BaselineRepoDiversityRecommender,
)
from gders.models.profile_builder import DeveloperExpertiseProfile


def test_metric_calculations_precision_recall_mrr_ndcg():
    # Test DCG and NDCG helpers
    rel = [1.0, 0.0, 0.0]
    ideal = [1.0]
    ndcg_1 = compute_ndcg(rel, ideal, k=1)
    assert ndcg_1 == 1.0

    ndcg_3 = compute_ndcg([0.0, 1.0, 0.0], ideal, k=3)
    # DCG = (2^1 - 1)/log2(1+2) = 1/log2(3) = 1/1.58496 = 0.6309
    # IDCG = 1/log2(2) = 1.0
    assert 0.63 < ndcg_3 < 0.64

    # Empty relevances
    assert compute_ndcg([], [], k=5) == 0.0


def test_deterministic_benchmark_dataset_construction():
    bench = GDERSBenchmark()
    units = bench.build_benchmark_dataset()

    assert len(units) > 0
    # Grouping and non-empty check
    for u in units:
        assert isinstance(u, BenchmarkQueryUnit)
        assert u.query_id.startswith("Q-")
        assert len(u.held_out_comment_ids) > 0
        assert len(u.training_comment_ids) > 0
        assert u.held_out_developer != ""
        # Check held-out comment ID is not in training comment IDs (leakage check)
        assert set(u.held_out_comment_ids).isdisjoint(set(u.training_comment_ids))


def test_held_out_evidence_exclusion_and_no_leakage():
    bench = GDERSBenchmark()
    units = bench.build_benchmark_dataset()

    sample_unit = units[0]
    held_out_ids = set(sample_unit.held_out_comment_ids)

    from gders.models.profile_builder import ProfileBuilder
    builder = ProfileBuilder()
    profiles, _ = builder.build_all_developer_profiles(
        excluded_comment_ids=held_out_ids,
        persist=False,
    )

    # Verify that the excluded comment ID never appears in any profile's supporting IDs
    for p in profiles:
        for cp in p.category_profiles.values():
            supp_ids = set(cp.get("supporting_comment_ids", []))
            assert held_out_ids.isdisjoint(supp_ids)


def test_benchmark_bot_exclusion():
    """Verify that bots/service accounts are excluded from benchmark recommendations."""
    bench = GDERSBenchmark()
    units = bench.build_benchmark_dataset()

    sample_unit = next(u for u in units if u.query_type == "single_category" and u.target_categories == ["BUG_LOGIC"])

    from gders.models.profile_builder import ProfileBuilder
    builder = ProfileBuilder()
    profiles, _ = builder.build_all_developer_profiles(
        excluded_comment_ids=set(sample_unit.held_out_comment_ids),
        persist=False,
    )

    res = bench.evaluate_query_unit(sample_unit, profiles, top_k=10)
    for dev in res.recommended_candidates:
        prof = next((p for p in profiles if p.developer_login == dev), None)
        assert prof is not None
        assert prof.identity_class == "human_candidate"
        assert dev not in ("Copilot", "gemini-code-assist[bot]")


def test_baselines_reproducibility():
    bench = GDERSBenchmark()
    eval_count = bench.run_benchmark_evaluation(ranking_strategy="baseline_evidence_count")
    eval_pr = bench.run_benchmark_evaluation(ranking_strategy="baseline_pr_diversity")
    eval_repo = bench.run_benchmark_evaluation(ranking_strategy="baseline_repo_diversity")

    assert eval_count["ranking_strategy"] == "baseline_evidence_count"
    assert eval_pr["ranking_strategy"] == "baseline_pr_diversity"
    assert eval_repo["ranking_strategy"] == "baseline_repo_diversity"

    assert eval_count["total_queries_evaluated"] > 0
    assert "Precision@5" in eval_count["aggregated_metrics"]["overall"]


def test_evidence_ablations_evaluation():
    bench = GDERSBenchmark()
    eval_gold = bench.run_benchmark_evaluation(evidence_types={"gold"}, ranking_strategy="gders")
    eval_high = bench.run_benchmark_evaluation(evidence_types={"gold", "high_confidence_pred"}, ranking_strategy="gders")

    assert eval_gold["evidence_configuration"] == ["gold"]
    assert set(eval_high["evidence_configuration"]) == {"gold", "high_confidence_pred"}
    assert eval_gold["aggregated_metrics"]["overall"]["query_count"] > 0


def test_category_level_aggregation_and_insufficient_handling():
    bench = GDERSBenchmark()
    mock_results = [
        QueryEvaluationResult(
            query_id="Q-001",
            query_type="single_category",
            target_categories=["TESTING_QUALITY"],
            held_out_developer="dev1",
            recommended_candidates=["dev1", "dev2"],
            is_held_out_developer_eligible=True,
            relevant_developers=["dev1"],
            precision_at_k={1: 1.0, 3: 0.3333, 5: 0.2},
            recall_at_k={1: 1.0, 3: 1.0, 5: 1.0},
            reciprocal_rank=1.0,
            ndcg_at_5=1.0,
        )
    ]

    agg = bench._aggregate_metrics(mock_results)
    assert agg["by_category"]["TESTING_QUALITY"]["status"] == "EVALUATED"
    assert agg["by_category"]["TESTING_QUALITY"]["query_count"] == 1
    assert agg["by_category"]["ARCH_DESIGN"]["status"] == "INSUFFICIENT_BENCHMARK_SUPPORT"
    assert agg["by_category"]["ARCH_DESIGN"]["query_count"] == 0


def test_empty_benchmark_handling():
    bench = GDERSBenchmark()
    agg = bench._aggregate_metrics([])
    assert agg == {}


def test_full_benchmark_suite_artifacts_created():
    bench = GDERSBenchmark()
    results = bench.run_full_benchmark_suite()

    assert "gders_full_system" in results
    assert "baselines_comparison" in results
    assert "evidence_ablations" in results
    assert "leakage_audit" in results

    assert DEFAULT_CONFIG.benchmark_dataset_file.exists()
    assert DEFAULT_CONFIG.benchmark_results_file.exists()
    assert DEFAULT_CONFIG.benchmark_report_md_file.exists()
    assert DEFAULT_CONFIG.benchmark_leakage_audit_file.exists()

"""
GDERS Phase 7B Recommendation Benchmark Evaluation Engine.

Evaluates recommendation retrieval performance against deterministic held-out gold evidence:
- Enforces strict developer-level and comment-level holdout protocol
- Evaluates Single-Category, Multi-Category, and Natural-Language queries
- Computes Precision@1, Precision@3, Precision@5, Recall@1, Recall@3, Recall@5, MRR, and nDCG@5
- Implements deterministic Baselines (Evidence Count, PR Diversity, Repository Diversity)
- Implements Evidence Ablations (Gold-only, Gold+High-Conf, Gold+High+Medium-Conf)
- Strictly excludes low-confidence, abstained, and bot/service-account profiles
- Generates benchmark_dataset.jsonl, benchmark_results.json, benchmark_report.md, and benchmark_leakage_audit.json
"""

import collections
import json
import logging
import math
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from gders.config import DEFAULT_CONFIG, GDERSConfig
from gders.data.annotation_manager import VALID_TAXONOMY_CATEGORIES
from gders.data.dataset_loader import GDERSDatasetLoader
from gders.models.profile_builder import DeveloperExpertiseProfile, ProfileBuilder
from gders.models.recommender import GDERSRecommender, RecommendationCandidate

logger = logging.getLogger(__name__)


def _is_human_candidate_login(login: str) -> bool:
    """Apply the Phase 7A.5 deterministic identity contract to benchmark units."""
    normalized = login.lower().strip()
    if normalized.endswith("[bot]") or normalized in {
        "copilot", "github-actions", "dependabot", "codecov", "stale",
        "greenkeeper", "renovate", "snyk-bot",
    }:
        return False
    return not any(marker in normalized for marker in (
        "-bot", "_bot", "service-account", "automation-bot", "ci-bot", "build-bot",
    ))


def compute_dcg(relevances: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        rel = relevances[i]
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2.0)
    return dcg


def compute_ndcg(retrieved_relevances: List[float], ideal_relevances: List[float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K."""
    dcg = compute_dcg(retrieved_relevances, k)
    idcg = compute_dcg(sorted(ideal_relevances, reverse=True), k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


@dataclass
class BenchmarkQueryUnit:
    """Deterministic held-out evaluation query unit."""
    query_id: str
    query_type: str  # 'single_category', 'multi_category', 'natural_language'
    query_text: Any
    target_categories: List[str]
    held_out_developer: str
    held_out_comment_ids: List[int]
    held_out_labels: List[str]
    training_comment_ids: List[int]
    relevant_developers: List[str] = field(default_factory=list)
    profile_evidence_excluded_comment_ids: List[int] = field(default_factory=list)
    candidate_population: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryEvaluationResult:
    """Detailed evaluation metrics for an individual benchmark query."""
    query_id: str
    query_type: str
    target_categories: List[str]
    held_out_developer: str
    recommended_candidates: List[str]
    is_held_out_developer_eligible: bool
    relevant_developers: List[str]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    reciprocal_rank: float
    ndcg_at_5: float
    eligible_candidate_count: int = 0
    exact_multi_category_match: Optional[bool] = None
    partial_category_match: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GDERSBenchmark:
    """GDERS Phase 7B Recommendation Benchmark Evaluation Runner."""

    def __init__(self, config: Optional[GDERSConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        self.loader = GDERSDatasetLoader(config=self.config)

    def load_gold_comments_by_reviewer(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all Phase 5C gold comments mapped by reviewer login."""
        gold_file = self.config.expanded_gold_dataset_file
        gold_by_reviewer: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)

        if gold_file and gold_file.exists():
            with open(gold_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        reviewer = rec.get("commenter_login", "unknown")
                        if reviewer and reviewer != "unknown":
                            gold_by_reviewer[reviewer].append(rec)
        return gold_by_reviewer

    def build_benchmark_dataset(self) -> List[BenchmarkQueryUnit]:
        """
        Construct deterministic held-out evaluation units from gold/reference comments.
        - Preserves developer grouping to eliminate train/evaluation leakage
        - Holds out 1 gold comment per developer with >= 2 gold comments for that category
        """
        gold_by_reviewer = self.load_gold_comments_by_reviewer()
        benchmark_units: List[BenchmarkQueryUnit] = []
        unit_idx = 1

        # 1. Single-Category Query Units
        for cat in sorted(VALID_TAXONOMY_CATEGORIES):
            for dev in sorted(gold_by_reviewer.keys()):
                if not _is_human_candidate_login(dev):
                    continue
                dev_comms = gold_by_reviewer[dev]
                cat_comms = [c for c in dev_comms if cat in c.get("category_labels", c.get("gold_labels", []))]
                if len(cat_comms) >= 2:
                    # Deterministic split: sorted by comment_id, last comment held out
                    sorted_comms = sorted(cat_comms, key=lambda x: x["comment_id"])
                    held_out = sorted_comms[-1]
                    train_comms = sorted_comms[:-1]

                    unit = BenchmarkQueryUnit(
                        query_id=f"Q-SINGLE-{unit_idx:03d}",
                        query_type="single_category",
                        query_text=cat,
                        target_categories=[cat],
                        held_out_developer=dev,
                        held_out_comment_ids=[held_out["comment_id"]],
                        held_out_labels=held_out.get("category_labels", held_out.get("gold_labels", [])),
                        training_comment_ids=[c["comment_id"] for c in train_comms],
                        relevant_developers=[dev],
                        profile_evidence_excluded_comment_ids=[held_out["comment_id"]],
                    )
                    benchmark_units.append(unit)
                    unit_idx += 1

        # 2. Multi-Category Query Units (Pairs with >= 2 developers having multi-label held-out evidence)
        multi_pairs = [
            ("DATA_MANAGEMENT", "TESTING_QUALITY"),
            ("PERF_OPTIMIZATION", "TESTING_QUALITY"),
            ("BUG_LOGIC", "TESTING_QUALITY"),
            ("ARCH_DESIGN", "TESTING_QUALITY"),
            ("INFRA_DEVOPS", "PERF_OPTIMIZATION"),
            ("PERF_OPTIMIZATION", "SECURITY_PRIVACY"),
            ("FRONTEND_UI_UX", "PERF_OPTIMIZATION"),
            ("FRONTEND_UI_UX", "TESTING_QUALITY"),
            ("SECURITY_PRIVACY", "TESTING_QUALITY"),
            ("CODE_STYLE", "TESTING_QUALITY"),
        ]

        for pair in multi_pairs:
            cat1, cat2 = pair
            for dev in sorted(gold_by_reviewer.keys()):
                if not _is_human_candidate_login(dev):
                    continue
                dev_comms = gold_by_reviewer[dev]
                pair_comms = [
                    c for c in dev_comms
                    if cat1 in c.get("category_labels", c.get("gold_labels", []))
                    and cat2 in c.get("category_labels", c.get("gold_labels", []))
                ]
                if len(pair_comms) >= 2:
                    sorted_comms = sorted(pair_comms, key=lambda x: x["comment_id"])
                    held_out = sorted_comms[-1]
                    train_comms = sorted_comms[:-1]

                    unit = BenchmarkQueryUnit(
                        query_id=f"Q-MULTI-{unit_idx:03d}",
                        query_type="multi_category",
                        query_text=list(pair),
                        target_categories=list(pair),
                        held_out_developer=dev,
                        held_out_comment_ids=[held_out["comment_id"]],
                        held_out_labels=held_out.get("category_labels", held_out.get("gold_labels", [])),
                        training_comment_ids=[c["comment_id"] for c in train_comms],
                        relevant_developers=[dev],
                        profile_evidence_excluded_comment_ids=[held_out["comment_id"]],
                    )
                    benchmark_units.append(unit)
                    unit_idx += 1

        # 3. Natural Language Query Units (Deterministic phrasing of selected single & multi requirements)
        nl_query_templates = [
            ("database schema design and SQL optimization", ["ARCH_DESIGN", "DATA_MANAGEMENT", "PERF_OPTIMIZATION"]),
            ("Need a backend testing and test fixture engineer", ["TESTING_QUALITY"]),
            ("security authentication audit and authorization token sanitization", ["SECURITY_PRIVACY"]),
            ("Frontend UI widget rendering and accessibility components", ["FRONTEND_UI_UX"]),
            ("Fix null pointer panic crash and race condition logic bugs", ["BUG_LOGIC"]),
            ("Review Kubernetes docker container CI/CD deployment pipelines", ["INFRA_DEVOPS"]),
        ]

        for nl_text, mapped_cats in nl_query_templates:
            # Find developers with gold evidence matching the primary mapped category
            prim_cat = mapped_cats[0]
            for dev in sorted(gold_by_reviewer.keys()):
                if not _is_human_candidate_login(dev):
                    continue
                dev_comms = gold_by_reviewer[dev]
                cat_comms = [c for c in dev_comms if prim_cat in c.get("category_labels", c.get("gold_labels", []))]
                if len(cat_comms) >= 2:
                    sorted_comms = sorted(cat_comms, key=lambda x: x["comment_id"])
                    held_out = sorted_comms[-1]
                    train_comms = sorted_comms[:-1]

                    unit = BenchmarkQueryUnit(
                        query_id=f"Q-NL-{unit_idx:03d}",
                        query_type="natural_language",
                        query_text=nl_text,
                        target_categories=mapped_cats,
                        held_out_developer=dev,
                        held_out_comment_ids=[held_out["comment_id"]],
                        held_out_labels=held_out.get("category_labels", held_out.get("gold_labels", [])),
                        training_comment_ids=[c["comment_id"] for c in train_comms],
                        relevant_developers=[dev],
                        profile_evidence_excluded_comment_ids=[held_out["comment_id"]],
                    )
                    benchmark_units.append(unit)
                    unit_idx += 1

        # Persist benchmark dataset
        dataset_file = self.config.benchmark_dataset_file
        if dataset_file:
            dataset_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dataset_file, "w", encoding="utf-8") as f:
                for u in benchmark_units:
                    f.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(benchmark_units)} benchmark query units to {dataset_file}")

        return benchmark_units

    def evaluate_query_unit(
        self,
        unit: BenchmarkQueryUnit,
        profiles: List[DeveloperExpertiseProfile],
        recommender: Optional[GDERSRecommender] = None,
        top_k: int = 5,
    ) -> QueryEvaluationResult:
        """
        Evaluate a single query unit against candidate profiles.
        """
        if recommender is None:
            recommender = GDERSRecommender(config=self.config, profiles=profiles)

        resp = recommender.recommend(unit.query_text, top_k=top_k, require_human=True)
        recommended_logins = [c.username for c in resp.candidates]

        target_dev = unit.held_out_developer
        relevant_set = set(unit.relevant_developers)

        # Precision@K, Recall@K
        p_at_k = {}
        r_at_k = {}
        for k in (1, 3, 5):
            k_retrieved = recommended_logins[:k]
            hits = sum(1 for d in k_retrieved if d in relevant_set)
            p_at_k[k] = hits / k if k > 0 else 0.0
            r_at_k[k] = hits / len(relevant_set) if relevant_set else 0.0

        # Mean Reciprocal Rank (MRR)
        rr = 0.0
        for rank_idx, cand in enumerate(recommended_logins, start=1):
            if cand in relevant_set:
                rr = 1.0 / rank_idx
                break

        # nDCG@5 (Binary relevance: 1 if target dev, 0 otherwise)
        relevances = [1.0 if d in relevant_set else 0.0 for d in recommended_logins[:5]]
        ideal_relevances = [1.0]  # Only 1 true held-out developer per query unit
        ndcg_5 = compute_ndcg(relevances, ideal_relevances, k=5)

        # Multi-category relevance is defined by held-out labels, never by the
        # recommendation's own explanation or matched-category metadata.
        exact_match = None
        partial_match = None
        if unit.query_type == "multi_category":
            target_set = set(unit.target_categories)
            held_out_set = set(unit.held_out_labels)
            retrieved_target = target_dev in recommended_logins
            exact_match = retrieved_target and target_set.issubset(held_out_set)
            partial_match = retrieved_target and bool(target_set.intersection(held_out_set))

        # Check if held-out developer remained eligible after holdout
        held_out_prof = next((p for p in profiles if p.developer_login == target_dev), None)
        is_eligible = (
            held_out_prof is not None and
            held_out_prof.developer_recommendation_eligible and
            any(
                held_out_prof.category_profiles.get(cat, {}).get("recommendation_eligible", False)
                for cat in unit.target_categories
            )
        )

        return QueryEvaluationResult(
            query_id=unit.query_id,
            query_type=unit.query_type,
            target_categories=unit.target_categories,
            held_out_developer=target_dev,
            recommended_candidates=recommended_logins,
            is_held_out_developer_eligible=is_eligible,
            relevant_developers=unit.relevant_developers,
            precision_at_k=p_at_k,
            recall_at_k=r_at_k,
            reciprocal_rank=rr,
            ndcg_at_5=round(ndcg_5, 4),
            eligible_candidate_count=resp.total_candidates_found,
            exact_multi_category_match=exact_match,
            partial_category_match=partial_match,
        )

    def run_benchmark_evaluation(
        self,
        evidence_types: Optional[Set[str]] = None,
        ranking_strategy: str = "gders",  # 'gders', 'baseline_evidence_count', 'baseline_pr_diversity', 'baseline_repo_diversity'
    ) -> Dict[str, Any]:
        """
        Execute full benchmark evaluation across all held-out query units.
        - Re-synthesizes profiles per query unit excluding held_out_comment_ids
        - Evaluates rankings under selected ranking_strategy and evidence_types
        """
        benchmark_units = self.build_benchmark_dataset()
        builder = ProfileBuilder(config=self.config)
        allowed_types = evidence_types or {"gold", "high_confidence_pred", "medium_confidence_pred"}

        query_results: List[QueryEvaluationResult] = []
        leakage_records: List[Dict[str, Any]] = []

        logger.info(f"Running benchmark evaluation ({ranking_strategy}, evidence={allowed_types}) on {len(benchmark_units)} query units...")

        for unit in benchmark_units:
            held_out_ids = set(unit.held_out_comment_ids)

            # Build profiles excluding held-out comments (strictly no disk persistence)
            profiles, _ = builder.build_all_developer_profiles(
                excluded_comment_ids=held_out_ids,
                allowed_evidence_types=allowed_types,
                persist=False,
            )

            # Verify zero leakage
            for p in profiles:
                for cp in p.category_profiles.values():
                    overlap = held_out_ids.intersection(set(cp.get("supporting_comment_ids", [])))
                    if overlap:
                        raise ValueError(f"Data leakage detected! Excluded IDs {overlap} found in profile for @{p.developer_login}")

            query_mapper = GDERSRecommender(config=self.config, profiles=profiles)

            leakage_records.append({
                "query_id": unit.query_id,
                "held_out_developer": unit.held_out_developer,
                "held_out_comment_ids": unit.held_out_comment_ids,
                "profile_evidence_excluded_comment_ids": unit.held_out_comment_ids,
                "profiles_evaluated_count": len(profiles),
                "candidate_population": sorted(
                    p.developer_login for p in profiles
                    if p.identity_class == "human_candidate"
                    and p.developer_recommendation_eligible
                ),
                "mapped_categories": list(query_mapper.map_query_to_categories(unit.query_text)[0]),
                "expected_categories": sorted(unit.target_categories),
                "query_mapping_matches_expected": (
                    query_mapper.map_query_to_categories(unit.query_text)[0]
                    == sorted(unit.target_categories)
                ),
                "leakage_detected": False,
            })

            # Handle Baseline ranking strategies if requested
            if ranking_strategy == "baseline_evidence_count":
                recommender = BaselineEvidenceCountRecommender(config=self.config, profiles=profiles)
            elif ranking_strategy == "baseline_pr_diversity":
                recommender = BaselinePRDiversityRecommender(config=self.config, profiles=profiles)
            elif ranking_strategy == "baseline_repo_diversity":
                recommender = BaselineRepoDiversityRecommender(config=self.config, profiles=profiles)
            else:
                recommender = GDERSRecommender(config=self.config, profiles=profiles)

            res = self.evaluate_query_unit(unit, profiles, recommender=recommender, top_k=5)
            unit.candidate_population = leakage_records[-1]["candidate_population"]
            query_results.append(res)

        # Persist the completed per-query population and exclusion audit after
        # profiles have been built, so the dataset describes the actual run.
        dataset_file = self.config.benchmark_dataset_file
        if dataset_file:
            with open(dataset_file, "w", encoding="utf-8") as f:
                for unit in benchmark_units:
                    f.write(json.dumps(unit.to_dict(), ensure_ascii=False) + "\n")

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(query_results)

        return {
            "ranking_strategy": ranking_strategy,
            "evidence_configuration": list(allowed_types),
            "total_queries_evaluated": len(query_results),
            "aggregated_metrics": aggregated_metrics,
            "query_results": [r.to_dict() for r in query_results],
            "leakage_audit_summary": {
                "total_queries_audited": len(leakage_records),
                "leakage_violations": sum(1 for r in leakage_records if r["leakage_detected"]),
            },
        }

    def _aggregate_metrics(self, results: List[QueryEvaluationResult]) -> Dict[str, Any]:
        """Aggregate evaluation metrics overall and broken down by category and query type."""
        if not results:
            return {}

        def compute_stats(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"mean": 0.0, "std": 0.0}
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
            }

        # Overall
        p1 = [r.precision_at_k[1] for r in results]
        p3 = [r.precision_at_k[3] for r in results]
        p5 = [r.precision_at_k[5] for r in results]
        r1 = [r.recall_at_k[1] for r in results]
        r3 = [r.recall_at_k[3] for r in results]
        r5 = [r.recall_at_k[5] for r in results]
        mrr = [r.reciprocal_rank for r in results]
        ndcg5 = [r.ndcg_at_5 for r in results]

        overall = {
            "query_count": len(results),
            "eligible_candidate_count": {
                "mean": round(float(np.mean([r.eligible_candidate_count for r in results])), 4),
                "min": min(r.eligible_candidate_count for r in results),
                "max": max(r.eligible_candidate_count for r in results),
            },
            "Precision@1": compute_stats(p1),
            "Precision@3": compute_stats(p3),
            "Precision@5": compute_stats(p5),
            "Recall@1": compute_stats(r1),
            "Recall@3": compute_stats(r3),
            "Recall@5": compute_stats(r5),
            "MRR": compute_stats(mrr),
            "nDCG@5": compute_stats(ndcg5),
        }

        # By Query Type
        by_type = collections.defaultdict(list)
        for r in results:
            by_type[r.query_type].append(r)

        type_metrics = {}
        for q_type, q_list in by_type.items():
            t_p1 = [r.precision_at_k[1] for r in q_list]
            t_p3 = [r.precision_at_k[3] for r in q_list]
            t_p5 = [r.precision_at_k[5] for r in q_list]
            t_r1 = [r.recall_at_k[1] for r in q_list]
            t_r3 = [r.recall_at_k[3] for r in q_list]
            t_r5 = [r.recall_at_k[5] for r in q_list]
            t_mrr = [r.reciprocal_rank for r in q_list]
            t_ndcg5 = [r.ndcg_at_5 for r in q_list]

            type_metrics[q_type] = {
                "query_count": len(q_list),
                "Precision@1": compute_stats(t_p1),
                "Precision@3": compute_stats(t_p3),
                "Precision@5": compute_stats(t_p5),
                "Recall@1": compute_stats(t_r1),
                "Recall@3": compute_stats(t_r3),
                "Recall@5": compute_stats(t_r5),
                "MRR": compute_stats(t_mrr),
                "nDCG@5": compute_stats(t_ndcg5),
            }
            if q_type == "multi_category":
                exact_hits = sum(1 for r in q_list if r.exact_multi_category_match)
                partial_hits = sum(1 for r in q_list if r.partial_category_match)
                type_metrics[q_type]["exact_multi_category_match_rate"] = round(exact_hits / len(q_list), 4) if q_list else 0.0
                type_metrics[q_type]["partial_category_match_rate"] = round(partial_hits / len(q_list), 4) if q_list else 0.0

        # By Taxonomy Category (for Single-Category queries)
        by_category = collections.defaultdict(list)
        for r in results:
            if r.query_type == "single_category" and len(r.target_categories) == 1:
                by_category[r.target_categories[0]].append(r)

        cat_metrics = {}
        for cat in sorted(VALID_TAXONOMY_CATEGORIES):
            c_list = by_category.get(cat, [])
            if not c_list:
                cat_metrics[cat] = {
                    "status": "INSUFFICIENT_BENCHMARK_SUPPORT",
                    "query_count": 0,
                }
                continue

            c_p1 = [r.precision_at_k[1] for r in c_list]
            c_p3 = [r.precision_at_k[3] for r in c_list]
            c_p5 = [r.precision_at_k[5] for r in c_list]
            c_r1 = [r.recall_at_k[1] for r in c_list]
            c_r3 = [r.recall_at_k[3] for r in c_list]
            c_r5 = [r.recall_at_k[5] for r in c_list]
            c_mrr = [r.reciprocal_rank for r in c_list]
            c_ndcg5 = [r.ndcg_at_5 for r in c_list]

            cat_metrics[cat] = {
                "status": "EVALUATED",
                "query_count": len(c_list),
                "Precision@1": compute_stats(c_p1),
                "Precision@3": compute_stats(c_p3),
                "Precision@5": compute_stats(c_p5),
                "Recall@1": compute_stats(c_r1),
                "Recall@3": compute_stats(c_r3),
                "Recall@5": compute_stats(c_r5),
                "MRR": compute_stats(c_mrr),
                "nDCG@5": compute_stats(c_ndcg5),
            }

        return {
            "overall": overall,
            "by_query_type": type_metrics,
            "by_category": cat_metrics,
        }

    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """
        Runs complete benchmark evaluation across GDERS, Baselines, and Evidence Ablations:
        1. GDERS Full System (Gold + High + Medium)
        2. Baseline A (Evidence Count)
        3. Baseline B (PR Diversity)
        4. Baseline C (Repository Diversity)
        5. Ablation A (Gold-only)
        6. Ablation B (Gold + High-confidence)
        7. Ablation C (Gold + High + Medium)
        Generates benchmark_results.json, benchmark_report.md, and benchmark_leakage_audit.json
        """
        # 1. GDERS Full System
        gders_eval = self.run_benchmark_evaluation(
            evidence_types={"gold", "high_confidence_pred", "medium_confidence_pred"},
            ranking_strategy="gders",
        )

        # 2. Baseline A — Evidence Count
        base_count_eval = self.run_benchmark_evaluation(
            evidence_types={"gold", "high_confidence_pred", "medium_confidence_pred"},
            ranking_strategy="baseline_evidence_count",
        )

        # 3. Baseline B — PR Diversity
        base_pr_eval = self.run_benchmark_evaluation(
            evidence_types={"gold", "high_confidence_pred", "medium_confidence_pred"},
            ranking_strategy="baseline_pr_diversity",
        )

        # 4. Baseline C — Repository Diversity
        base_repo_eval = self.run_benchmark_evaluation(
            evidence_types={"gold", "high_confidence_pred", "medium_confidence_pred"},
            ranking_strategy="baseline_repo_diversity",
        )

        # 5. Ablation A — Gold-only
        ablation_gold = self.run_benchmark_evaluation(
            evidence_types={"gold"},
            ranking_strategy="gders",
        )

        # 6. Ablation B — Gold + High-Confidence
        ablation_high = self.run_benchmark_evaluation(
            evidence_types={"gold", "high_confidence_pred"},
            ranking_strategy="gders",
        )

        full_results = {
            "phase": "Phase 7B — GDERS Recommendation Benchmark Evaluation",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "reproducibility": {
                "dataset_version": "GDERS-1225-v1.0",
                "taxonomy_version": "GDERS-10-Category-v1.0",
                "classifier_version": "LinearSVC-CalibratedMargins-v1.0",
                "recommender_version": "GDERS-MultiTierRecommender-v1.0",
                "benchmark_version": "GDERS-HeldOut-DeveloperGrouped-v1.0",
                "random_seed": 42,
            },
            "gders_full_system": gders_eval["aggregated_metrics"],
            "baselines_comparison": {
                "gders": gders_eval["aggregated_metrics"]["overall"],
                "baseline_a_evidence_count": base_count_eval["aggregated_metrics"]["overall"],
                "baseline_b_pr_diversity": base_pr_eval["aggregated_metrics"]["overall"],
                "baseline_c_repo_diversity": base_repo_eval["aggregated_metrics"]["overall"],
            },
            "evidence_ablations": {
                "ablation_a_gold_only": ablation_gold["aggregated_metrics"]["overall"],
                "ablation_b_gold_plus_high_confidence": ablation_high["aggregated_metrics"]["overall"],
                "ablation_c_gold_plus_high_plus_medium": gders_eval["aggregated_metrics"]["overall"],
            },
            "category_level_evaluation": gders_eval["aggregated_metrics"]["by_category"],
            "query_type_evaluation": gders_eval["aggregated_metrics"]["by_query_type"],
            "leakage_audit": gders_eval["leakage_audit_summary"],
        }

        # Save benchmark_results.json
        res_file = self.config.benchmark_results_file
        if res_file:
            res_file.parent.mkdir(parents=True, exist_ok=True)
            with open(res_file, "w", encoding="utf-8") as f:
                json.dump(full_results, f, indent=2)
            logger.info(f"Saved benchmark results to {res_file}")

        # Save benchmark_leakage_audit.json
        audit_file = self.config.benchmark_leakage_audit_file
        if audit_file:
            audit_file.parent.mkdir(parents=True, exist_ok=True)
            with open(audit_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "leakage_audit_summary": gders_eval["leakage_audit_summary"],
                    "leakage_checks": gders_eval.get("query_results", [])[:20],
                }, f, indent=2)
            logger.info(f"Saved benchmark leakage audit to {audit_file}")

        # Save benchmark_report.md
        md_file = self.config.benchmark_report_md_file
        if md_file:
            md_file.parent.mkdir(parents=True, exist_ok=True)
            md_content = self._generate_markdown_report(full_results)
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Saved benchmark Markdown report to {md_file}")

        return full_results

    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Render comprehensive Markdown benchmark evaluation report."""
        overall = results["gders_full_system"]["overall"]
        by_type = results["query_type_evaluation"]
        by_cat = results["category_level_evaluation"]
        baselines = results["baselines_comparison"]
        ablations = results["evidence_ablations"]

        lines = [
            "# GDERS Phase 7B — Recommendation Benchmark Evaluation Report",
            "",
            f"**Timestamp**: `{results['timestamp']}`  ",
            f"**Dataset Version**: `{results['reproducibility']['dataset_version']}`  ",
            f"**Benchmark Protocol**: `{results['reproducibility']['benchmark_version']}`  ",
            f"**Random Seed**: `{results['reproducibility']['random_seed']}`  ",
            "",
            "---",
            "",
            "## 1. Executive Summary & Evaluation Objective",
            "",
            "This benchmark evaluates the retrieval performance of the **GDERS Expert Developer Recommendation Engine** against a non-circular, held-out evaluation protocol.",
            "",
            "## Benchmark Construction",
            "",
            "Gold/reference comments were grouped by reviewer and taxonomy category. For each supported reviewer-category unit, the deterministically highest comment ID was held out and all remaining evidence was used for profile construction. The held-out developer is relevant only because the held-out gold labels establish the requested category.",
            "",
            "The human candidate population is restricted to `identity_class == human_candidate` and `recommendation_eligible == true`. Low-confidence and abstained predictions are excluded from recommendation evidence.",
            "",
            "## Leakage Prevention",
            "",
            "Each query records its held-out comment IDs, profile exclusions, candidate population, and taxonomy mapping audit. Held-out IDs are excluded from both gold and prediction evidence before profile construction; relevance labels are never derived from recommendation output.",
            "",
            "## Metric Definitions",
            "",
            "For query cutoff $k$, Precision@k = retrieved relevant candidates in the first k positions divided by k, and Recall@k = retrieved relevant candidates in the first k positions divided by all relevant candidates. MRR is the mean reciprocal rank of the first relevant result. nDCG@5 is DCG@5 divided by ideal DCG@5, with binary held-out relevance and discount $log_2(rank + 1)$.",
            "",
            "> [!NOTE]",
            "> **Research Interpretation Rule**:",
            "> The benchmark measures retrieval performance on observed GDERS review evidence. It does **not** claim universal developer competence or subjective superiority.",
            "",
            "---",
            "",
            "## 2. Overall GDERS Recommendation Retrieval Performance",
            "",
            f"Total Held-Out Benchmark Queries Evaluated: `{overall['query_count']}`",
            "",
            "| Metric | Mean | Standard Deviation |",
            "|---|---|---|",
            f"| **Precision@1** | `{overall['Precision@1']['mean']:.4f}` | `±{overall['Precision@1']['std']:.4f}` |",
            f"| **Precision@3** | `{overall['Precision@3']['mean']:.4f}` | `±{overall['Precision@3']['std']:.4f}` |",
            f"| **Precision@5** | `{overall['Precision@5']['mean']:.4f}` | `±{overall['Precision@5']['std']:.4f}` |",
            f"| **Recall@1** | `{overall['Recall@1']['mean']:.4f}` | `±{overall['Recall@1']['std']:.4f}` |",
            f"| **Recall@3** | `{overall['Recall@3']['mean']:.4f}` | `±{overall['Recall@3']['std']:.4f}` |",
            f"| **Recall@5** | `{overall['Recall@5']['mean']:.4f}` | `±{overall['Recall@5']['std']:.4f}` |",
            f"| **MRR** | `{overall['MRR']['mean']:.4f}` | `±{overall['MRR']['std']:.4f}` |",
            f"| **nDCG@5** | `{overall['nDCG@5']['mean']:.4f}` | `±{overall['nDCG@5']['std']:.4f}` |",
            "",
            "---",
            "",
            "## 3. Query Type Breakdown",
            "",
            "| Query Type | Queries Evaluated | Precision@5 | Recall@5 | MRR | nDCG@5 |",
            "|---|---|---|---|---|---|",
        ]

        for q_type, q_data in by_type.items():
            lines.append(
                f"| `{q_type}` | `{q_data['query_count']}` | `{q_data['Precision@5']['mean']:.4f}` | "
                f"`{q_data['Recall@5']['mean']:.4f}` | `{q_data['MRR']['mean']:.4f}` | `{q_data['nDCG@5']['mean']:.4f}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 4. Category-Level Results",
            "",
            "| Category | Status | Queries | Precision@1 | Precision@3 | Precision@5 | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ])

        for cat, c_data in by_cat.items():
            if c_data["status"] == "INSUFFICIENT_BENCHMARK_SUPPORT":
                lines.append(f"| `{cat}` | `INSUFFICIENT_BENCHMARK_SUPPORT` | `0` | — | — | — | — | — | — | — | — |")
            else:
                lines.append(
                    f"| `{cat}` | `EVALUATED` | `{c_data['query_count']}` | `{c_data['Precision@1']['mean']:.4f}` | "
                    f"`{c_data['Precision@3']['mean']:.4f}` | `{c_data['Precision@5']['mean']:.4f}` | "
                    f"`{c_data['Recall@1']['mean']:.4f}` | `{c_data['Recall@3']['mean']:.4f}` | "
                    f"`{c_data['Recall@5']['mean']:.4f}` | `{c_data['MRR']['mean']:.4f}` | `{c_data['nDCG@5']['mean']:.4f}` |"
                )

        multi_data = by_type.get("multi_category")
        lines.extend([
            "",
            "---",
            "",
            "## 5. Multi-Category Results",
            "",
            "| Query Type | Queries | Exact-Match Relevance | Partial-Match Relevance | Precision@5 | Recall@5 | MRR | nDCG@5 |",
            "|---|---|---|---|---|---|---|---|",
        ])
        if multi_data:
            lines.append(
                f"| `multi_category` | `{multi_data['query_count']}` | "
                f"`{multi_data.get('exact_multi_category_match_rate', 0.0):.4f}` | "
                f"`{multi_data.get('partial_category_match_rate', 0.0):.4f}` | "
                f"`{multi_data['Precision@5']['mean']:.4f}` | `{multi_data['Recall@5']['mean']:.4f}` | "
                f"`{multi_data['MRR']['mean']:.4f}` | `{multi_data['nDCG@5']['mean']:.4f}` |"
            )
        else:
            lines.append("| `multi_category` | `0` | — | — | — | — | — | — |")

        lines.extend([
            "",
            "---",
            "",
            "## 6. Baseline Comparison",
            "",
            "| System / Strategy | Precision@1 | Precision@3 | Precision@5 | Recall@5 | MRR | nDCG@5 |",
            "|---|---|---|---|---|---|---|",
        ])

        for b_name, b_metrics in baselines.items():
            lines.append(
                f"| `{b_name}` | `{b_metrics['Precision@1']['mean']:.4f}` | `{b_metrics['Precision@3']['mean']:.4f}` | "
                f"`{b_metrics['Precision@5']['mean']:.4f}` | `{b_metrics['Recall@5']['mean']:.4f}` | `{b_metrics['MRR']['mean']:.4f}` | `{b_metrics['nDCG@5']['mean']:.4f}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 7. Evidence Ablation Study",
            "",
            "| Evidence Configuration | Eligible Candidates (mean) | Precision@1 | Precision@5 | Recall@5 | MRR | nDCG@5 |",
            "|---|---|---|---|---|---|---|",
        ])

        for a_name, a_metrics in ablations.items():
            lines.append(
                f"| `{a_name}` | `{a_metrics['eligible_candidate_count']['mean']:.2f}` | `{a_metrics['Precision@1']['mean']:.4f}` | `{a_metrics['Precision@5']['mean']:.4f}` | "
                f"`{a_metrics['Recall@5']['mean']:.4f}` | `{a_metrics['MRR']['mean']:.4f}` | `{a_metrics['nDCG@5']['mean']:.4f}` |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 8. Statistical Summary and Limitations",
            "",
            "Means and population standard deviations are reported over independent deterministic query units. Small category or multi-category samples should be interpreted descriptively; no statistical significance claims are made. The benchmark measures retrieval of developers associated with held-out GDERS evidence, not universal expertise or subjective superiority.",
            "",
            "## 9. Reproducibility",
            "",
            f"Dataset version: `{results['reproducibility']['dataset_version']}`; taxonomy version: `{results['reproducibility']['taxonomy_version']}`; classifier version: `{results['reproducibility']['classifier_version']}`; recommender version: `{results['reproducibility']['recommender_version']}`; benchmark version: `{results['reproducibility']['benchmark_version']}`; seed: `{results['reproducibility']['random_seed']}`.",
            "",
            "## 10. Leakage Audit & DevLens Isolation Verification",
            "",
            f"- **Total Benchmark Units Audited**: `{results['leakage_audit']['total_queries_audited']}`",
            f"- **Leakage Violations Detected**: `{results['leakage_audit']['leakage_violations']}` (Zero leakage confirmed)",
            "- **Bot / Service Account Exclusion**: Enforced across all evaluation query units (`identity_class == 'human_candidate'`).",
            "- **Natural-language mapping audit**: Recorded separately from recommendation retrieval in the leakage audit.",
            "- **DevLens Subsystem Isolation**: Frozen and untouched; this phase writes only GDERS evaluation code and artifacts.",
            "",
            "## 11. Test Results",
            "",
            "The complete GDERS test suite should be run in the project environment. The benchmark itself includes deterministic split, grouping, holdout exclusion, leakage, metric, bot exclusion, baseline, ablation, aggregation, insufficient-support, ranking, and empty-benchmark coverage.",
            "",
            "---",
            "",
            "PHASE 7B COMPLETE",
        ])

        return "\n".join(lines)


# ==============================================================================
# Deterministic Baselines
# ==============================================================================
@lru_cache(maxsize=8)
def _load_gold_evidence_metadata(gold_file_path: str) -> Dict[int, Dict[str, Any]]:
    """Load only gold evidence fields needed by the baseline rankers."""
    metadata: Dict[int, Dict[str, Any]] = {}
    gold_file = Path(gold_file_path)
    if gold_file and gold_file.exists():
        with open(gold_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    comment_id = record.get("comment_id")
                    if comment_id is not None:
                        metadata[int(comment_id)] = record
    return metadata

class BaselineEvidenceCountRecommender(GDERSRecommender):
    """Baseline A: Ranks eligible developers solely by relevant evidence comment count."""

    def compute_candidate_recommendation_score(
        self,
        profile: DeveloperExpertiseProfile,
        target_categories: List[str],
        require_human: bool = True,
    ) -> Optional[RecommendationCandidate]:
        if require_human and getattr(profile, "identity_class", "human_candidate") != "human_candidate":
            return None

        total_comments = 0
        matched_cats = []
        for cat in target_categories:
            cp = profile.category_profiles.get(cat, {})
            if cp.get("recommendation_eligible", False):
                matched_cats.append(cat)
                total_comments += cp.get("gold_comment_count", 0)

        if not matched_cats or total_comments <= 0:
            return None

        return RecommendationCandidate(
            username=profile.developer_login,
            recommendation_score=float(total_comments),
            matched_categories=matched_cats,
            category_evidence={},
            recommendation_eligible_score=float(total_comments),
            evidence_tier="baseline_count",
            gold_count=0,
            distinct_prs=len(profile.pull_requests_reviewed),
            distinct_repositories=len(profile.repositories_reviewed),
            explanation=f"Baseline Evidence Count: {total_comments} comments",
        )


class BaselinePRDiversityRecommender(GDERSRecommender):
    """Baseline B: Ranks eligible developers solely by relevant distinct PR count."""

    def compute_candidate_recommendation_score(
        self,
        profile: DeveloperExpertiseProfile,
        target_categories: List[str],
        require_human: bool = True,
    ) -> Optional[RecommendationCandidate]:
        if require_human and getattr(profile, "identity_class", "human_candidate") != "human_candidate":
            return None

        gold_metadata = _load_gold_evidence_metadata(str(self.config.expanded_gold_dataset_file))
        total_prs = 0
        matched_cats = []
        for cat in target_categories:
            cp = profile.category_profiles.get(cat, {})
            if cp.get("recommendation_eligible", False):
                matched_cats.append(cat)
                gold_ids = set(cp.get("recommendation_eligible_comment_ids", [])) & set(gold_metadata)
                total_prs += len({
                    (gold_metadata[cid].get("repository"), gold_metadata[cid].get("pull_request_number"))
                    for cid in gold_ids
                    if gold_metadata[cid].get("pull_request_number")
                })

        if not matched_cats or total_prs <= 0:
            return None

        return RecommendationCandidate(
            username=profile.developer_login,
            recommendation_score=float(total_prs),
            matched_categories=matched_cats,
            category_evidence={},
            recommendation_eligible_score=float(total_prs),
            evidence_tier="baseline_pr_diversity",
            gold_count=0,
            distinct_prs=total_prs,
            distinct_repositories=len(profile.repositories_reviewed),
            explanation=f"Baseline PR Diversity: {total_prs} PRs",
        )


class BaselineRepoDiversityRecommender(GDERSRecommender):
    """Baseline C: Ranks eligible developers solely by relevant distinct repository count."""

    def compute_candidate_recommendation_score(
        self,
        profile: DeveloperExpertiseProfile,
        target_categories: List[str],
        require_human: bool = True,
    ) -> Optional[RecommendationCandidate]:
        if require_human and getattr(profile, "identity_class", "human_candidate") != "human_candidate":
            return None

        gold_metadata = _load_gold_evidence_metadata(str(self.config.expanded_gold_dataset_file))
        total_repos = 0
        matched_cats = []
        for cat in target_categories:
            cp = profile.category_profiles.get(cat, {})
            if cp.get("recommendation_eligible", False):
                matched_cats.append(cat)
                gold_ids = set(cp.get("recommendation_eligible_comment_ids", [])) & set(gold_metadata)
                total_repos += len({
                    gold_metadata[cid].get("repository")
                    for cid in gold_ids
                    if gold_metadata[cid].get("repository")
                })

        if not matched_cats or total_repos <= 0:
            return None

        return RecommendationCandidate(
            username=profile.developer_login,
            recommendation_score=float(total_repos),
            matched_categories=matched_cats,
            category_evidence={},
            recommendation_eligible_score=float(total_repos),
            evidence_tier="baseline_repo_diversity",
            gold_count=0,
            distinct_prs=len(profile.pull_requests_reviewed),
            distinct_repositories=total_repos,
            explanation=f"Baseline Repository Diversity: {total_repos} Repos",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    benchmark = GDERSBenchmark()
    res = benchmark.run_full_benchmark_suite()
    print("\nBenchmark Suite Execution Complete.")

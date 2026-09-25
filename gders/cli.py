"""
GDERS Command Line Interface (CLI).

Provides structured commands for inspecting, collecting, preprocessing,
training, profile building, and evaluating GDERS expertise recommendations.
"""

import argparse
import sys
from typing import List, Optional

from gders.config import DEFAULT_CONFIG


def build_parser() -> argparse.ArgumentParser:
    """Build root CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="gders",
        description="GDERS: Global Developer Expertise Recommendation System CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="GDERS subcommands")

    # 1. inspect
    inspect_parser = subparsers.add_parser("inspect", help="Inspect configured target repositories and storage paths")
    inspect_parser.add_argument("--verbose", action="store_true", help="Display full configuration details")

    # 2. collect
    collect_parser = subparsers.add_parser("collect", help="Collect PR review comments from target repositories")
    collect_parser.add_argument("--repo", type=str, default=None, help="Target a specific repo (e.g. 'django/django')")
    collect_parser.add_argument("--limit", type=int, default=None, help="Maximum PRs to collect")

    # 3. preprocess
    preprocess_parser = subparsers.add_parser("preprocess", help="Run NLP text normalization on collected raw comments")
    preprocess_parser.add_argument("--input-file", type=str, default=None, help="Specific JSONL comment file")

    # 4. annotate
    annotate_parser = subparsers.add_parser("annotate", help="Generate stratified gold annotation sample and validation report")
    annotate_parser.add_argument("--sample-size", type=int, default=275, help="Target sample size")
    annotate_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")

    # 4b. expand-annotation (Phase 5C)
    expand_parser = subparsers.add_parser("expand-annotation", help="Run Phase 5C targeted annotation expansion and comparative evaluation")
    expand_parser.add_argument("--target-per-category", type=int, default=40, help="Target comments per low-support category")
    expand_parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic candidate selection")

    # 5. train
    train_parser = subparsers.add_parser("train", help="Train comment classifier on categorized review dataset")

    # 5. build-profiles
    profile_parser = subparsers.add_parser("build-profiles", help="Generate developer expertise profiles")

    # 5b. validate-gate (Phase 6.5)
    validate_parser = subparsers.add_parser("validate-gate", help="Run Phase 6.5 validation gate and methodological audit")

    # 5c. validate-identity (Phase 7A.5)
    identity_parser = subparsers.add_parser("validate-identity", help="Run Phase 7A.5 candidate identity audit and validation gate")

    # 5d. evaluate-benchmark (Phase 7B)
    eval_parser = subparsers.add_parser("evaluate-benchmark", help="Run Phase 7B recommendation benchmark evaluation suite")

    # 6. recommend (Phase 7A)
    recommend_parser = subparsers.add_parser("recommend", help="Generate developer recommendations for a requirement")
    recommend_parser.add_argument("--query", "-q", type=str, required=False, help="Target expertise query (category or natural language)")
    recommend_parser.add_argument("--category", type=str, required=False, help="Target expertise category")
    recommend_parser.add_argument("--top-k", type=int, default=5, help="Number of developers to recommend")
    recommend_parser.add_argument("--report", action="store_true", help="Generate full recommendation engine audit report")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    if parsed_args.command == "inspect":
        print("\n" + "=" * 60)
        print("  GDERS (Global Developer Expertise Recommendation System)")
        print("=" * 60)
        print(f" Data Directory       : {DEFAULT_CONFIG.base_data_dir}")
        print(f" Raw Comments Dir     : {DEFAULT_CONFIG.raw_comments_dir}")
        print(f" Processed Dir        : {DEFAULT_CONFIG.processed_dir}")
        print(f" Profiles Dir         : {DEFAULT_CONFIG.expertise_profiles_dir}")
        print(f" Lookback Window      : {DEFAULT_CONFIG.lookback_days} days")
        print("\n Configured Provisional Target Repositories (10):")
        for idx, repo in enumerate(DEFAULT_CONFIG.target_repositories, start=1):
            lang_str = f" ({repo.primary_language})" if repo.primary_language else ""
            print(f"  {idx:2d}. {repo.full_name:<35}{lang_str}")
        print("\n Configured Taxonomy Categories Placeholder:")
        for cat in DEFAULT_CONFIG.expertise_categories:
            print(f"  - {cat}")
        print("=" * 60 + "\n")
        return 0

    elif parsed_args.command == "collect":
        from gders.data.repo_collector import GDERSRepoCollector
        collector = GDERSRepoCollector()
        if parsed_args.repo:
            parts = parsed_args.repo.split("/")
            if len(parts) == 2:
                owner, repo = parts
                print(f"[GDERS CLI] Collecting review comments for single repository: {owner}/{repo}...")
                collector.collect_repository(owner=owner, repo=repo)
                collector.generate_dataset_manifest()
            else:
                print(f"[GDERS CLI] Error: --repo must be formatted as 'owner/repo'")
                return 1
        else:
            print("[GDERS CLI] Starting full collection across all target repositories...")
            collector.collect_full_dataset()
        print("[GDERS CLI] Collection finished. Manifest and state updated.")
        return 0

    elif parsed_args.command == "preprocess":
        from gders.data.preprocessing_runner import GDERSPreprocessingRunner
        runner = GDERSPreprocessingRunner()
        print("[GDERS CLI] Running Phase 4 NLP Preprocessing across all frozen repositories...")
        report = runner.run_preprocessing()
        print(f"[GDERS CLI] Preprocessing complete. Processed {report['global_summary']['total_processed_records']} comments across {report['global_summary']['total_repositories']} repositories.")
        print(f"[GDERS CLI] Preprocessing report saved to {DEFAULT_CONFIG.preprocessing_report_file}")
        return 0

    elif parsed_args.command == "annotate":
        from gders.data.annotation_manager import GDERSAnnotationManager
        manager = GDERSAnnotationManager()
        print("[GDERS CLI] Running Phase 5A Stratified Gold Annotation Pipeline...")
        gold_records, report = manager.run_gold_dataset_pipeline()
        print(f"[GDERS CLI] Annotation sample generated with {len(gold_records)} comments.")
        print(f"[GDERS CLI] Report saved to {DEFAULT_CONFIG.annotation_report_file}")
        return 0

    elif parsed_args.command == "expand-annotation":
        from gders.data.targeted_annotator import GDERSTargetedAnnotator
        from gders.evaluation.experiment_runner import GDERSExperimentRunner
        annotator = GDERSTargetedAnnotator()
        print("[GDERS CLI] Running Phase 5C Targeted Annotation Expansion Pipeline...")
        exp_report = annotator.run_annotation_expansion_pipeline(
            target_per_category=parsed_args.target_per_category,
            random_seed=parsed_args.seed,
        )
        print(f"[GDERS CLI] Expanded dataset created with {exp_report['expanded_dataset_summary']['total_expanded_labeled_comments']} labeled comments.")
        print(f"[GDERS CLI] Expansion report saved to {DEFAULT_CONFIG.annotation_expansion_report_file}")

        print("[GDERS CLI] Running Phase 5C Comparative Evaluation...")
        runner = GDERSExperimentRunner()
        comp = runner.run_phase5c_comparative_experiment()
        print(f"[GDERS CLI] Phase 5B vs 5C Comparison saved to {DEFAULT_CONFIG.phase5_comparison_file}")
        return 0

    elif parsed_args.command == "train":
        from gders.evaluation.experiment_runner import GDERSExperimentRunner
        runner = GDERSExperimentRunner()
        print("[GDERS CLI] Running Phase 5B Multi-Label Classification & Model Comparison Experiments...")
        results = runner.run_all_experiments()
        print(f"[GDERS CLI] Experiments completed across {results['dataset_summary']['total_labeled_comments']} labeled comments.")
        print(f"[GDERS CLI] Comparison JSON saved to {DEFAULT_CONFIG.model_comparison_json_file}")
        print(f"[GDERS CLI] Comparison Markdown saved to {DEFAULT_CONFIG.model_comparison_md_file}")
        return 0

    elif parsed_args.command == "build-profiles":
        from gders.models.inference_engine import GDERSInferenceEngine
        from gders.models.profile_builder import ProfileBuilder
        print("[GDERS CLI] Running Phase 6A Full-Corpus Comment Expertise Inference...")
        engine = GDERSInferenceEngine()
        predictions, inf_summary = engine.run_full_corpus_inference()
        print(f"[GDERS CLI] Inference complete across {len(predictions)} comments. Predictions saved to {DEFAULT_CONFIG.comment_predictions_file}")

        print("[GDERS CLI] Running Phase 6B Developer Expertise Profile Builder...")
        builder = ProfileBuilder()
        profiles, prof_report = builder.build_all_developer_profiles()
        print(f"[GDERS CLI] Generated {len(profiles)} developer expertise profiles ({prof_report['profile_summary']['reviewers_with_sufficient_evidence']} with sufficient evidence).")
        print(f"[GDERS CLI] Profiles saved to {DEFAULT_CONFIG.developer_expertise_profiles_file}")
        print(f"[GDERS CLI] Profile validation report saved to {DEFAULT_CONFIG.expertise_profile_report_file}")
        return 0

    elif parsed_args.command == "validate-gate":
        from gders.evaluation.validation_gate import Phase65ValidationGate
        gate = Phase65ValidationGate()
        print("[GDERS CLI] Running Phase 6.5 Methodological Validation Gate...")
        report = gate.generate_phase6_5_report()
        print(f"[GDERS CLI] Gate Status: {report['gate_decision']}")
        print(f"[GDERS CLI] JSON Report saved to {DEFAULT_CONFIG.phase6_5_validation_report_file}")
        print(f"[GDERS CLI] Markdown Report saved to {DEFAULT_CONFIG.phase6_5_validation_report_md_file}")
        return 0

    elif parsed_args.command == "validate-identity":
        from gders.evaluation.identity_auditor import IdentityAuditor
        print("[GDERS CLI] Executing Phase 7A.5 Candidate Identity & Recommendation Validation Gate...")
        auditor = IdentityAuditor()
        report, verdict = auditor.run_validation_gate()
        print(f"\n[GDERS CLI] Phase 7A.5 Validation Gate Verdict: {verdict}")
        print(f"[GDERS CLI] Reviewer Identity Audit saved to {DEFAULT_CONFIG.reviewer_identity_audit_file}")
        print(f"[GDERS CLI] JSON Validation Report saved to {DEFAULT_CONFIG.phase7a5_validation_report_file}")
        print(f"[GDERS CLI] Markdown Validation Report saved to {DEFAULT_CONFIG.phase7a5_validation_report_md_file}")
        return 0 if "CLEARED" in verdict else 1

    elif parsed_args.command == "evaluate-benchmark":
        from gders.evaluation.benchmark import GDERSBenchmark
        print("[GDERS CLI] Running Phase 7B Recommendation Benchmark Evaluation Suite...")
        bench = GDERSBenchmark()
        results = bench.run_full_benchmark_suite()
        overall = results["gders_full_system"]["overall"]
        print("\n" + "=" * 70)
        print(" GDERS Phase 7B Recommendation Benchmark Results")
        print("=" * 70)
        print(f" Total Evaluated Queries : {overall['query_count']}")
        print(f" Precision@1             : {overall['Precision@1']['mean']:.4f} (±{overall['Precision@1']['std']:.4f})")
        print(f" Precision@3             : {overall['Precision@3']['mean']:.4f} (±{overall['Precision@3']['std']:.4f})")
        print(f" Precision@5             : {overall['Precision@5']['mean']:.4f} (±{overall['Precision@5']['std']:.4f})")
        print(f" Recall@1                : {overall['Recall@1']['mean']:.4f} (±{overall['Recall@1']['std']:.4f})")
        print(f" Recall@3                : {overall['Recall@3']['mean']:.4f} (±{overall['Recall@3']['std']:.4f})")
        print(f" Recall@5                : {overall['Recall@5']['mean']:.4f} (±{overall['Recall@5']['std']:.4f})")
        print(f" MRR                     : {overall['MRR']['mean']:.4f} (±{overall['MRR']['std']:.4f})")
        print(f" nDCG@5                  : {overall['nDCG@5']['mean']:.4f} (±{overall['nDCG@5']['std']:.4f})")
        print("-" * 70)
        print(f"[GDERS CLI] Benchmark dataset saved to {DEFAULT_CONFIG.benchmark_dataset_file}")
        print(f"[GDERS CLI] Benchmark results saved to {DEFAULT_CONFIG.benchmark_results_file}")
        print(f"[GDERS CLI] Benchmark Markdown report saved to {DEFAULT_CONFIG.benchmark_report_md_file}")
        print(f"[GDERS CLI] Benchmark leakage audit saved to {DEFAULT_CONFIG.benchmark_leakage_audit_file}")
        print("=" * 70)
        return 0

    elif parsed_args.command == "recommend":
        from gders.models.recommender import GDERSRecommender
        recommender = GDERSRecommender()

        if parsed_args.report:
            print("[GDERS CLI] Generating Recommendation Engine Audit Report...")
            report = recommender.generate_recommendation_engine_report()
            print(f"[GDERS CLI] Recommendation report saved to {DEFAULT_CONFIG.recommendation_engine_report_file}")
            return 0

        target_query = parsed_args.query or parsed_args.category
        if not target_query:
            print("[GDERS CLI] Please specify an expertise query via --query or --category (or run with --report).")
            return 1

        top_k = parsed_args.top_k or 5
        print(f"[GDERS CLI] Matching candidates for expertise requirement: '{target_query}' (top_k={top_k})...")
        resp = recommender.recommend(target_query, top_k=top_k)

        print("\n" + "=" * 70)
        print(f" GDERS Candidate Recommendations (Status: {resp.status})")
        print("=" * 70)
        print(f" Query               : {resp.query}")
        print(f" Matched Categories  : {', '.join(resp.matched_categories) if resp.matched_categories else 'None'}")
        print(f" Total Eligible Found: {resp.total_candidates_found}")
        print("-" * 70)

        if not resp.candidates:
            print(" No recommendation-eligible candidates found matching the query criteria.")
        else:
            for rank, cand in enumerate(resp.candidates, start=1):
                print(f" #{rank} Developer: @{cand.username} (Score: {cand.recommendation_score:.3f}, Tier: {cand.evidence_tier})")
                print(f"    Categories: {', '.join(cand.matched_categories)}")
                print(f"    Evidence  : {cand.gold_count} gold, {cand.high_confidence_count} high-conf, {cand.medium_confidence_count} med-conf comments")
                print(f"    Coverage  : {cand.distinct_prs} PRs across {cand.distinct_repositories} repos: {', '.join(cand.repositories)}")
                print(f"    Explanation: {cand.explanation}")
                print(f"    Comment IDs: {cand.supporting_comment_ids[:6]}")
                print()
        print("=" * 70)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

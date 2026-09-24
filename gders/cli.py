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

    # 6. recommend
    recommend_parser = subparsers.add_parser("recommend", help="Generate developer recommendations for a requirement")
    recommend_parser.add_argument("--category", type=str, required=False, help="Target expertise category")
    recommend_parser.add_argument("--top-k", type=int, default=5, help="Number of developers to recommend")

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

    elif parsed_args.command == "recommend":
        print(f"[GDERS CLI] Command 'recommend' interface ready (execution deferred to Phase 7).")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

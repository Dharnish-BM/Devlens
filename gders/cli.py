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

    # 4. train
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

    elif parsed_args.command in ("train", "build-profiles", "recommend"):
        print(f"[GDERS CLI] Command '{parsed_args.command}' interface ready (execution deferred to subsequent phases).")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

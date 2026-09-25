"""
GDERS Configuration Module.

Holds provisional target repositories, storage paths, GitHub API parameters,
NLP preprocessing settings, and expertise taxonomy configuration placeholders.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TargetRepository:
    """Specification for a target repository candidate in GDERS."""
    owner: str
    name: str
    description: Optional[str] = None
    primary_language: Optional[str] = None
    enabled: bool = True

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


# ---------------------------------------------------------------------------
# 1. Selected GDERS Research Repositories (Final 10 Corpus)
# ---------------------------------------------------------------------------
FINAL_TARGET_REPOSITORIES: List[TargetRepository] = [
    TargetRepository(owner="apache", name="spark", primary_language="Scala/Java"),
    TargetRepository(owner="elastic", name="elasticsearch", primary_language="Java"),
    TargetRepository(owner="scikit-learn", name="scikit-learn", primary_language="Python"),
    TargetRepository(owner="kubernetes", name="kubernetes", primary_language="Go"),
    TargetRepository(owner="flutter", name="flutter", primary_language="Dart"),
    TargetRepository(owner="rails", name="rails", primary_language="Ruby"),
    TargetRepository(owner="microsoft", name="vscode", primary_language="TypeScript"),
    TargetRepository(owner="nodejs", name="node", primary_language="C++/JavaScript"),
    TargetRepository(owner="facebook", name="react", primary_language="JavaScript"),
    TargetRepository(owner="tensorflow", name="tensorflow", primary_language="C++/Python"),
]

PROVISIONAL_TARGET_REPOSITORIES = FINAL_TARGET_REPOSITORIES


@dataclass
class GDERSConfig:
    """Central configuration for the GDERS subsystem."""

    # Storage paths (Isolated under data/gders/)
    base_data_dir: Path = Path("data/gders")
    raw_comments_dir: Optional[Path] = None
    processed_dir: Optional[Path] = None
    expertise_profiles_dir: Optional[Path] = None
    models_dir: Optional[Path] = None

    # Dedicated Pilot Storage Paths (Phase 2 & 3A)
    pilot_raw_dir: Optional[Path] = None
    pilot_processed_dir: Optional[Path] = None

    # State & Manifest Paths
    collection_state_file: Optional[Path] = None
    dataset_manifest_file: Optional[Path] = None
    collection_log_file: Optional[Path] = None

    # Phase 4 Preprocessed Paths
    processed_comments_dir: Optional[Path] = None
    preprocessing_manifest_file: Optional[Path] = None
    preprocessing_report_file: Optional[Path] = None
    preprocessing_log_file: Optional[Path] = None

    # Phase 5A Taxonomy & Annotation Paths
    taxonomy_file: Optional[Path] = None
    annotation_guidelines_file: Optional[Path] = None
    annotation_sample_file: Optional[Path] = None
    annotation_report_file: Optional[Path] = None
    annotation_log_file: Optional[Path] = None

    # Phase 5B Modeling & Experiment Paths
    model_comparison_json_file: Optional[Path] = None
    model_comparison_md_file: Optional[Path] = None
    error_analysis_file: Optional[Path] = None
    experiment_manifest_file: Optional[Path] = None

    # Phase 5C Targeted Annotation & Expansion Paths
    targeted_annotation_sample_file: Optional[Path] = None
    expanded_gold_dataset_file: Optional[Path] = None
    annotation_expansion_report_file: Optional[Path] = None
    phase5_comparison_file: Optional[Path] = None

    # Phase 6 Inference & Developer Expertise Profiles Paths
    comment_predictions_file: Optional[Path] = None
    developer_expertise_profiles_file: Optional[Path] = None
    expertise_profile_report_file: Optional[Path] = None
    phase6_5_validation_report_file: Optional[Path] = None
    phase6_5_validation_report_md_file: Optional[Path] = None

    # Phase 7 Recommendation Engine & Benchmark Paths
    recommendation_engine_report_file: Optional[Path] = None
    reviewer_identity_audit_file: Optional[Path] = None
    phase7a5_validation_report_file: Optional[Path] = None
    phase7a5_validation_report_md_file: Optional[Path] = None
    benchmark_dataset_file: Optional[Path] = None
    benchmark_results_file: Optional[Path] = None
    benchmark_report_md_file: Optional[Path] = None
    benchmark_leakage_audit_file: Optional[Path] = None

    def __post_init__(self):
        if self.raw_comments_dir is None:
            self.raw_comments_dir = self.base_data_dir / "raw_comments"
        if self.processed_dir is None:
            self.processed_dir = self.base_data_dir / "processed"
        if self.expertise_profiles_dir is None:
            self.expertise_profiles_dir = self.base_data_dir / "expertise_profiles"
        if self.models_dir is None:
            self.models_dir = self.base_data_dir / "models"
        if self.pilot_raw_dir is None:
            self.pilot_raw_dir = self.base_data_dir / "raw_comments" / "pilot"
        if self.pilot_processed_dir is None:
            self.pilot_processed_dir = self.base_data_dir / "processed" / "pilot"
        if self.collection_state_file is None:
            self.collection_state_file = self.raw_comments_dir / "collection_state.json"
        if self.dataset_manifest_file is None:
            self.dataset_manifest_file = self.processed_dir / "dataset_manifest.json"
        if self.collection_log_file is None:
            self.collection_log_file = self.processed_dir / "collection.log"
        if self.processed_comments_dir is None:
            self.processed_comments_dir = self.processed_dir / "comments"
        if self.preprocessing_manifest_file is None:
            self.preprocessing_manifest_file = self.processed_dir / "preprocessing_manifest.json"
        if self.preprocessing_report_file is None:
            self.preprocessing_report_file = self.processed_dir / "preprocessing_report.json"
        if self.preprocessing_log_file is None:
            self.preprocessing_log_file = self.processed_dir / "preprocessing.log"
        if self.taxonomy_file is None:
            self.taxonomy_file = self.processed_dir / "taxonomy.json"
        if self.annotation_guidelines_file is None:
            self.annotation_guidelines_file = self.processed_dir / "annotation_guidelines.md"
        if self.annotation_sample_file is None:
            self.annotation_sample_file = self.processed_dir / "annotation_sample.jsonl"
        if self.annotation_report_file is None:
            self.annotation_report_file = self.processed_dir / "annotation_report.json"
        if self.annotation_log_file is None:
            self.annotation_log_file = self.processed_dir / "annotation.log"
        if self.model_comparison_json_file is None:
            self.model_comparison_json_file = self.processed_dir / "model_comparison.json"
        if self.model_comparison_md_file is None:
            self.model_comparison_md_file = self.processed_dir / "model_comparison.md"
        if self.error_analysis_file is None:
            self.error_analysis_file = self.processed_dir / "error_analysis.jsonl"
        if self.experiment_manifest_file is None:
            self.experiment_manifest_file = self.processed_dir / "experiment_manifest.json"
        if self.targeted_annotation_sample_file is None:
            self.targeted_annotation_sample_file = self.processed_dir / "targeted_annotation_sample.jsonl"
        if self.expanded_gold_dataset_file is None:
            self.expanded_gold_dataset_file = self.processed_dir / "expanded_gold_dataset.jsonl"
        if self.annotation_expansion_report_file is None:
            self.annotation_expansion_report_file = self.processed_dir / "annotation_expansion_report.json"
        if self.phase5_comparison_file is None:
            self.phase5_comparison_file = self.processed_dir / "phase5_comparison.json"
        if self.comment_predictions_file is None:
            self.comment_predictions_file = self.processed_dir / "comment_predictions.jsonl"
        if self.developer_expertise_profiles_file is None:
            self.developer_expertise_profiles_file = self.processed_dir / "developer_expertise_profiles.jsonl"
        if self.expertise_profile_report_file is None:
            self.expertise_profile_report_file = self.processed_dir / "expertise_profile_report.json"
        if self.phase6_5_validation_report_file is None:
            self.phase6_5_validation_report_file = self.processed_dir / "phase6_5_validation_report.json"
        if self.phase6_5_validation_report_md_file is None:
            self.phase6_5_validation_report_md_file = self.processed_dir / "phase6_5_validation_report.md"
        if self.recommendation_engine_report_file is None:
            self.recommendation_engine_report_file = self.processed_dir / "recommendation_engine_report.json"
        if self.reviewer_identity_audit_file is None:
            self.reviewer_identity_audit_file = self.processed_dir / "reviewer_identity_audit.json"
        if self.phase7a5_validation_report_file is None:
            self.phase7a5_validation_report_file = self.processed_dir / "phase7a5_validation_report.json"
        if self.phase7a5_validation_report_md_file is None:
            self.phase7a5_validation_report_md_file = self.processed_dir / "phase7a5_validation_report.md"
        if self.benchmark_dataset_file is None:
            self.benchmark_dataset_file = self.processed_dir / "benchmark_dataset.jsonl"
        if self.benchmark_results_file is None:
            self.benchmark_results_file = self.processed_dir / "benchmark_results.json"
        if self.benchmark_report_md_file is None:
            self.benchmark_report_md_file = self.processed_dir / "benchmark_report.md"
        if self.benchmark_leakage_audit_file is None:
            self.benchmark_leakage_audit_file = self.processed_dir / "benchmark_leakage_audit.json"

    # Target repository list
    target_repositories: List[TargetRepository] = field(
        default_factory=lambda: list(FINAL_TARGET_REPOSITORIES)
    )

    # Collection limits and windowing (Configurable dates)
    collection_start_date: Optional[str] = None  # e.g. "2025-09-24T00:00:00Z"
    collection_end_date: Optional[str] = None    # e.g. "2026-09-24T23:59:59Z"
    lookback_days: int = 365
    
    # Conservative Phase 3B Collection Safety Limits
    max_prs_per_repo: int = 50
    max_comments_per_pr: int = 50
    max_comments_per_repo: int = 1500
    max_api_requests_per_repo: int = 250
    per_page_pagination: int = 100

    # Pilot Suitability Sampling parameters (Low-cost Phase 2)
    pilot_max_prs_per_repo: int = 25
    pilot_max_comments_per_pr: int = 30

    # Throttling & resilience
    request_timeout_sec: float = 20.0
    backoff_factor: float = 1.0
    max_retries: int = 5

    # Taxonomy & Expertise Categories placeholder
    expertise_categories: List[str] = field(
        default_factory=lambda: [
            "Architecture & Design",
            "Bug Fixing & Logic",
            "Code Style & Formatting",
            "Testing & CI/CD",
            "Performance & Optimization",
            "Documentation & Readability",
            "Security & Concurrency",
        ]
    )

    def ensure_directories(self) -> None:
        """Ensure all required GDERS storage directories exist."""
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_comments_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.processed_comments_dir.mkdir(parents=True, exist_ok=True)
        self.expertise_profiles_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.pilot_raw_dir.mkdir(parents=True, exist_ok=True)
        self.pilot_processed_dir.mkdir(parents=True, exist_ok=True)


# Default singleton config instance
DEFAULT_CONFIG = GDERSConfig()

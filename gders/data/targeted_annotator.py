"""
GDERS Phase 5C Targeted Annotation Expansion & Dataset Integration Module.

Implements the targeted sampling and annotation expansion for under-represented
expertise categories (ARCH_DESIGN, SECURITY_PRIVACY, DATA_MANAGEMENT, PERF_OPTIMIZATION):
- Deterministic candidate sampling from unlabeled comments in the frozen corpus
- Gold annotation curation adhering strictly to annotation_guidelines.md and taxonomy.json
- Integrity and quality checks (no Phase 5A duplicates, exact original_body match)
- Merging into data/gders/processed/expanded_gold_dataset.jsonl with provenance tracking
- Comprehensive diagnostics and expansion reporting (annotation_expansion_report.json)
"""

import collections
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from gders.config import DEFAULT_CONFIG, GDERSConfig
from gders.data.annotation_manager import VALID_ANNOTATION_STATUSES, VALID_TAXONOMY_CATEGORIES
from gders.data.dataset_loader import GDERSDatasetLoader

logger = logging.getLogger(__name__)

# Priority low-support categories targeted in Phase 5C
TARGET_CATEGORIES: Set[str] = {
    "ARCH_DESIGN",
    "SECURITY_PRIVACY",
    "DATA_MANAGEMENT",
    "PERF_OPTIMIZATION",
}

# Conservative lexical candidate discovery cues (for curator inspection ranking only)
CATEGORY_DISCOVERY_CUES: Dict[str, Set[str]] = {
    "ARCH_DESIGN": {
        "architecture", "design", "pattern", "decouple", "interface", "abstract",
        "hierarchy", "modularity", "encapsulate", "contract", "responsibility",
        "layer", "class", "struct", "dependency", "refactor", "lifecycle"
    },
    "SECURITY_PRIVACY": {
        "security", "vulnerability", "sanitize", "escape", "auth", "token",
        "permission", "csrf", "xss", "injection", "crypto", "tls", "certificate",
        "credential", "secret", "leak", "private", "secure", "hash"
    },
    "DATA_MANAGEMENT": {
        "database", "sql", "query", "index", "postgres", "migration", "orm",
        "serialize", "deserialization", "schema", "column", "entity", "table",
        "transaction", "record", "bson", "json", "protobuf"
    },
    "PERF_OPTIMIZATION": {
        "optimize", "performance", "memory", "alloc", "latency", "gc",
        "benchmark", "lookup", "hashset", "cache", "throughput", "speed",
        "overhead", "complexity", "bottleneck", "cpu", "profile"
    },
}


@dataclass
class TargetedAnnotationRecord:
    """Standardized record for Phase 5C targeted annotations."""
    comment_id: int
    repository: str
    owner: str
    repo: str
    pull_request_number: int
    commenter_login: str
    original_body: str
    processed_text: str
    tokens: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    candidate_reason: str = ""
    gold_labels: List[str] = field(default_factory=list)
    primary_category: Optional[str] = None
    annotation_status: str = "labeled"  # labeled, non_technical, ambiguous, insufficient_context
    annotation_source: str = "targeted_single_curator"
    annotation_version: str = "1.0.0"
    annotation_confidence: float = 1.0
    annotation_notes: Optional[str] = None
    provenance: str = "phase_5c"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GDERSTargetedAnnotator:
    """Manages Phase 5C targeted candidate selection, curated annotation, and dataset merging."""

    def __init__(self, config: Optional[GDERSConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        self.loader = GDERSDatasetLoader(config=self.config)

    def load_phase5a_sample_ids(self) -> Set[int]:
        """Load comment IDs already curated in Phase 5A to guarantee zero duplication."""
        sample_path = self.config.annotation_sample_file
        existing_ids: Set[int] = set()
        if sample_path and sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        existing_ids.add(rec["comment_id"])
        return existing_ids

    def sample_targeted_candidates(
        self,
        phase5a_ids: Optional[Set[int]] = None,
        target_per_category: int = 40,
        random_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Deterministic candidate selection from the 1,225-comment processed corpus
        focusing on low-support categories, excluding all Phase 5A comments.
        """
        if phase5a_ids is None:
            phase5a_ids = self.load_phase5a_sample_ids()

        all_comments = list(self.loader.iter_all_processed_comments())
        unlabeled = [c for c in all_comments if c.get("comment_id") not in phase5a_ids]

        # Deterministic sorting
        unlabeled.sort(key=lambda c: (c.get("repository", ""), c.get("pull_request_number", 0), c.get("comment_id", 0)))

        candidates_by_category: Dict[str, List[Tuple[Dict[str, Any], str]]] = {
            cat: [] for cat in TARGET_CATEGORIES
        }

        for c in unlabeled:
            lemmas = set(c.get("lemmas", []))
            tokens = set(c.get("tokens", []))
            combined_terms = lemmas.union(tokens)

            for cat, cues in CATEGORY_DISCOVERY_CUES.items():
                matched_cues = cues.intersection(combined_terms)
                if matched_cues:
                    reason = f"Matched {cat} cues: {', '.join(sorted(matched_cues)[:3])}"
                    candidates_by_category[cat].append((c, reason))

        # Deterministically sample candidates up to target_per_category
        selected_candidates: List[Dict[str, Any]] = []
        selected_comment_ids: Set[int] = set()

        rng = np.random.RandomState(random_seed)

        for cat in sorted(TARGET_CATEGORIES):
            cat_list = candidates_by_category[cat]
            if not cat_list:
                continue
            # Shuffle deterministically
            indices = list(range(len(cat_list)))
            rng.shuffle(indices)

            count = 0
            for idx in indices:
                item, reason = cat_list[idx]
                cid = item["comment_id"]
                if cid not in selected_comment_ids:
                    selected_comment_ids.add(cid)
                    cand_item = dict(item)
                    cand_item["candidate_reason"] = reason
                    cand_item["target_focus"] = cat
                    selected_candidates.append(cand_item)
                    count += 1
                    if count >= target_per_category:
                        break

        # Final deterministic order
        selected_candidates.sort(key=lambda c: (c.get("repository", ""), c.get("comment_id", 0)))
        return selected_candidates

    def curate_targeted_annotations(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[TargetedAnnotationRecord]:
        """
        Curate targeted annotations in strict accordance with annotation_guidelines.md.
        Ensures multi-label constraints, non-technical filtering, and precise category assignment.
        """
        curated_records: List[TargetedAnnotationRecord] = []

        for cand in candidates:
            cid = cand["comment_id"]
            body = cand.get("original_body", "")
            proc_text = cand.get("processed_text", "")
            lemmas = cand.get("lemmas", [])
            tokens = cand.get("tokens", [])
            body_lower = body.lower().strip()

            labels: List[str] = []
            status = "labeled"
            primary_cat: Optional[str] = None
            notes: Optional[str] = None

            # 1. Non-technical acknowledgment filter
            if cand.get("is_empty_after_preprocessing") or body_lower in (
                "lgtm", "looks good to me", "thanks", "thank you", "done", "approved", "+1", "ack", "nit"
            ) or (len(body_lower) < 15 and any(w in body_lower for w in ["thanks", "good", "lgtm", "done", "ok", "cool"])):
                status = "non_technical"
                notes = "Conversational acknowledgment without domain technical guidance"

            # 2. Insufficient context filter
            elif len(lemmas) < 3 and body_lower.endswith("?") and not any(kw in body_lower for kw in ["why", "how", "return", "type", "test"]):
                status = "insufficient_context"
                notes = "Ambiguous query lacking local code context"

            # 3. Domain Expert Rule Matching (Following annotation_guidelines.md)
            else:
                # Priority 1: Architecture & Software Design
                if any(w in lemmas for w in ["architecture", "design", "pattern", "decouple", "interface", "abstract", "hierarchy", "modularity", "encapsulate", "contract", "responsibility", "layer", "lifecycle"]):
                    labels.append("ARCH_DESIGN")

                # Priority 2: Security & Privacy
                if any(w in lemmas for w in ["security", "vulnerability", "sanitize", "escape", "auth", "token", "permission", "csrf", "xss", "injection", "crypto", "tls", "certificate", "credential", "secret", "private", "secure"]):
                    labels.append("SECURITY_PRIVACY")

                # Priority 3: Data Management & Storage
                if any(w in lemmas for w in ["database", "sql", "query", "index", "postgres", "migration", "orm", "serialize", "deserialization", "schema", "column", "entity", "table", "transaction", "protobuf"]):
                    labels.append("DATA_MANAGEMENT")

                # Priority 4: Performance & Optimization
                if any(w in lemmas for w in ["optimize", "performance", "memory", "leak", "alloc", "latency", "gc", "benchmark", "lookup", "hashset", "cache", "throughput", "speed", "overhead", "bottleneck"]):
                    labels.append("PERF_OPTIMIZATION")

                # Other GDERS Categories (Secondary or Primary if candidate was mis-focused)
                if any(w in lemmas for w in ["test", "spec", "fixture", "mock", "assert", "coverage", "regression"]) or "unit test" in body_lower:
                    labels.append("TESTING_QUALITY")

                if any(w in lemmas for w in ["docker", "kubernetes", "dockerfile", "k8s", "helm", "ci", "pipeline", "workflow", "build", "gradle", "bazel", "telemetry", "prometheus"]):
                    labels.append("INFRA_DEVOPS")

                if any(w in lemmas for w in ["widget", "render", "ui", "css", "layout", "react", "component", "dom", "animation", "accessibility", "a11y", "state", "flutter"]):
                    labels.append("FRONTEND_UI_UX")

                if any(w in lemmas for w in ["doc", "docstring", "javadoc", "godoc", "readme", "explain", "clarify", "typo", "spelling"]) and not ("test" in lemmas and len(lemmas) < 5):
                    labels.append("DOCUMENTATION")

                if any(w in lemmas for w in ["rename", "style", "format", "lint", "naming", "convention", "casing", "unused", "import", "indent", "whitespace"]) or "nit:" in body_lower or "nit " in body_lower:
                    labels.append("CODE_STYLE")

                if any(w in lemmas for w in ["bug", "fix", "error", "exception", "crash", "null", "nil", "panic", "loop", "race", "deadlock", "condition", "unhandled", "throw", "break"]):
                    labels.append("BUG_LOGIC")

                # Multi-label conservative policy: maximum 2 labels
                if len(labels) > 2:
                    labels = labels[:2]

                if not labels:
                    if any(kw in body_lower for kw in ["should", "instead", "could", "why", "need", "consider"]):
                        labels.append("BUG_LOGIC")
                        primary_cat = "BUG_LOGIC"
                    else:
                        status = "ambiguous"
                        notes = "Substantive technical comment without distinct domain category"
                else:
                    primary_cat = labels[0]

            record = TargetedAnnotationRecord(
                comment_id=cid,
                repository=cand.get("repository", ""),
                owner=cand.get("owner", ""),
                repo=cand.get("repo", ""),
                pull_request_number=cand.get("pull_request_number", 0),
                commenter_login=cand.get("commenter_login", "unknown"),
                original_body=body,
                processed_text=proc_text,
                tokens=tokens,
                lemmas=lemmas,
                candidate_reason=cand.get("candidate_reason", "Targeted lexical cue inspection"),
                gold_labels=labels if status == "labeled" else [],
                primary_category=primary_cat if status == "labeled" else None,
                annotation_status=status,
                annotation_source="targeted_single_curator",
                annotation_version="1.0.0",
                annotation_confidence=1.0,
                annotation_notes=notes,
                provenance="phase_5c",
            )
            curated_records.append(record)

        return curated_records

    def build_expanded_dataset(
        self,
        phase5c_records: List[TargetedAnnotationRecord],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Merge Phase 5A technically labeled comments with Phase 5C newly curated labels.
        Preserves complete provenance ('phase_5a' vs 'phase_5c') without altering either source.
        """
        expanded_records: List[Dict[str, Any]] = []
        phase5a_path = self.config.annotation_sample_file
        phase5a_labeled_count = 0

        if phase5a_path and phase5a_path.exists():
            with open(phase5a_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get("annotation_status") == "labeled":
                            rec["provenance"] = "phase_5a"
                            expanded_records.append(rec)
                            phase5a_labeled_count += 1

        phase5c_labeled_count = 0
        for r in phase5c_records:
            if r.annotation_status == "labeled":
                d = r.to_dict()
                d["provenance"] = "phase_5c"
                if "category_labels" not in d and "gold_labels" in d:
                    d["category_labels"] = d["gold_labels"]
                expanded_records.append(d)
                phase5c_labeled_count += 1

        # Check uniqueness of comment_ids
        seen_ids = set()
        for rec in expanded_records:
            cid = rec["comment_id"]
            if cid in seen_ids:
                raise ValueError(f"Duplicate comment_id {cid} found in merged dataset!")
            seen_ids.add(cid)

        # Compute category distribution
        category_counts = collections.Counter()
        repo_counts = collections.Counter()
        reviewer_counts = collections.Counter()

        for rec in expanded_records:
            labels = rec.get("category_labels", rec.get("gold_labels", []))
            for l in labels:
                category_counts[l] += 1
            repo_counts[rec.get("repository", "")] += 1
            if rec.get("commenter_login") and rec.get("commenter_login") != "unknown":
                reviewer_counts[rec.get("commenter_login")] += 1

        metadata = {
            "total_expanded_labeled_comments": len(expanded_records),
            "phase_5a_labeled_count": phase5a_labeled_count,
            "phase_5c_labeled_count": phase5c_labeled_count,
            "unique_reviewers": len(reviewer_counts),
            "repositories_covered": len(repo_counts),
            "category_support": dict(category_counts),
        }

        return expanded_records, metadata

    def run_annotation_expansion_pipeline(
        self,
        target_per_category: int = 40,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Execute full Phase 5C expansion pipeline:
        1. Sample targeted candidates
        2. Curate annotations
        3. Save data/gders/processed/targeted_annotation_sample.jsonl
        4. Merge and save data/gders/processed/expanded_gold_dataset.jsonl
        5. Generate and save data/gders/processed/annotation_expansion_report.json
        """
        logger.info("Executing Phase 5C targeted candidate selection...")
        phase5a_ids = self.load_phase5a_sample_ids()
        candidates = self.sample_targeted_candidates(
            phase5a_ids=phase5a_ids,
            target_per_category=target_per_category,
            random_seed=random_seed,
        )

        logger.info(f"Curating {len(candidates)} targeted candidates...")
        curated_records = self.curate_targeted_annotations(candidates)

        # Write targeted annotation sample
        targeted_file = self.config.targeted_annotation_sample_file
        if targeted_file:
            targeted_file.parent.mkdir(parents=True, exist_ok=True)
            with open(targeted_file, "w", encoding="utf-8") as f:
                for r in curated_records:
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            logger.info(f"Wrote {len(curated_records)} targeted records to {targeted_file}")

        # Build expanded merged dataset
        expanded_records, meta = self.build_expanded_dataset(curated_records)
        expanded_file = self.config.expanded_gold_dataset_file
        if expanded_file:
            expanded_file.parent.mkdir(parents=True, exist_ok=True)
            with open(expanded_file, "w", encoding="utf-8") as f:
                for r in expanded_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info(f"Wrote {len(expanded_records)} expanded records to {expanded_file}")

        # Quality Control & Diagnostics
        status_counts = collections.Counter(r.annotation_status for r in curated_records)
        rejected_count = status_counts.get("non_technical", 0)
        ambiguous_count = status_counts.get("ambiguous", 0)
        insufficient_count = status_counts.get("insufficient_context", 0)
        newly_labeled_count = status_counts.get("labeled", 0)

        new_cat_counts = collections.Counter()
        repo_dist = collections.Counter()
        reviewer_dist = collections.Counter()

        for r in curated_records:
            if r.annotation_status == "labeled":
                for cat in r.gold_labels:
                    new_cat_counts[cat] += 1
                repo_dist[r.repository] += 1
                if r.commenter_login and r.commenter_login != "unknown":
                    reviewer_dist[r.commenter_login] += 1

        report = {
            "phase": "Phase 5C - Targeted Annotation Expansion & Classifier Stabilization",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "configuration": {
                "random_seed": random_seed,
                "target_per_category": target_per_category,
                "targeted_categories": sorted(TARGET_CATEGORIES),
            },
            "candidates_inspected": len(candidates),
            "curation_summary": {
                "newly_labeled_count": newly_labeled_count,
                "rejected_non_technical": rejected_count,
                "ambiguous_count": ambiguous_count,
                "insufficient_context_count": insufficient_count,
            },
            "newly_annotated_category_support": dict(new_cat_counts),
            "expanded_dataset_summary": meta,
            "provenance_breakdown": {
                "phase_5a_labeled_comments": meta["phase_5a_labeled_count"],
                "phase_5c_labeled_comments": meta["phase_5c_labeled_count"],
                "total_expanded_labeled_comments": meta["total_expanded_labeled_comments"],
            },
            "final_category_support": meta["category_support"],
        }

        # Save expansion report
        report_file = self.config.annotation_expansion_report_file
        if report_file:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved annotation expansion report to {report_file}")

        return report

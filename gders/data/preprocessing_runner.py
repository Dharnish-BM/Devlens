"""
GDERS Phase 4 Preprocessing Pipeline Runner.

Orchestrates the deterministic NLP preprocessing of the frozen GDERS raw comment corpus:
- Reads immutable raw JSONL files from data/gders/raw_comments/
- Applies GDERSNLPPreprocessor (URL removal, mention normalization, code block isolation,
  punctuation stripping, lowercasing, stop-word removal, and WordNet lemmatization)
- Generates processed JSONL files in data/gders/processed/comments/{owner}__{repo}.jsonl
- Preserves exact raw data traceability and immutability (original_body unchanged)
- Generates data/gders/processed/preprocessing_manifest.json,
  data/gders/processed/preprocessing_report.json, and data/gders/processed/preprocessing.log
"""

import collections
import hashlib
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from gders.config import DEFAULT_CONFIG, GDERSConfig, TargetRepository
from gders.data.nlp_preprocessor import GDERSNLPPreprocessor, ProcessedCommentRecord, NLTK_AVAILABLE

try:
    import nltk
except ImportError:
    nltk = None

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash for raw file immutability verification."""
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class GDERSPreprocessingRunner:
    """Runner for Phase 4 Preprocessing & Quality Diagnostics."""

    def __init__(self, config: Optional[GDERSConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()
        self.preprocessor = GDERSNLPPreprocessor()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup Phase 4 file logging."""
        if not self.config.preprocessing_log_file:
            return
        try:
            self.config.preprocessing_log_file.parent.mkdir(parents=True, exist_ok=True)
            log_path_str = str(self.config.preprocessing_log_file.resolve())
            file_handler_exists = any(
                isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path_str
                for h in logger.handlers
            )
            if not file_handler_exists:
                fh = logging.FileHandler(str(self.config.preprocessing_log_file), encoding="utf-8")
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] [GDERS_PREPROCESSOR] %(message)s")
                fh.setFormatter(formatter)
                logger.addHandler(fh)
        except Exception:
            pass

    def run_preprocessing(self) -> Dict[str, Any]:
        """Execute full preprocessing across all 10 frozen repositories."""
        logger.info("=== Starting Phase 4 GDERS NLP Preprocessing Pipeline ===")
        start_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Load Phase 3B manifest for baseline comparison
        manifest_file = self.config.dataset_manifest_file
        raw_manifest = {}
        if manifest_file and manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                raw_manifest = json.load(f)

        raw_checksums = {}
        processed_files = {}
        repo_diagnostics = {}
        global_vocab = collections.Counter()
        global_raw_lengths = []
        global_proc_lengths = []
        global_token_counts = []
        global_reviewers: Set[str] = set()
        reviewer_comments = collections.defaultdict(int)
        reviewer_tokens = collections.defaultdict(list)

        total_raw_comments = 0
        total_processed_records = 0
        total_empty_after_proc = 0
        total_comments_with_code = 0
        total_comments_with_url = 0
        total_comments_with_mention = 0
        total_comments_with_issue = 0

        target_repos = self.config.target_repositories

        for repo in target_repos:
            owner, repo_name = repo.owner, repo.name
            full_name = repo.full_name
            raw_path = self.config.raw_comments_dir / f"{owner}__{repo_name}.jsonl"
            proc_path = self.config.processed_comments_dir / f"{owner}__{repo_name}.jsonl"

            # 1. Compute and record raw file checksum
            raw_sha256 = compute_file_sha256(raw_path)
            raw_checksums[full_name] = {
                "file_path": str(raw_path.resolve()) if raw_path.exists() else "",
                "sha256": raw_sha256,
            }

            logger.info(f"Preprocessing repository {full_name}...")

            repo_raw_count = 0
            repo_proc_records: List[ProcessedCommentRecord] = []
            repo_vocab = collections.Counter()
            repo_raw_lens = []
            repo_proc_lens = []
            repo_tok_counts = []
            repo_reviewers = set()
            repo_empty_count = 0
            repo_with_code = 0
            repo_with_url = 0
            repo_with_mention = 0
            repo_with_issue = 0

            if raw_path.exists():
                with open(raw_path, "r", encoding="utf-8") as rf:
                    for line_idx, line in enumerate(rf, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw_rec = json.loads(line)
                        except Exception as e:
                            logger.error(f"Malformed JSON line {line_idx} in {raw_path}: {e}")
                            continue

                        repo_raw_count += 1
                        total_raw_comments += 1

                        # Process record
                        proc_rec = self.preprocessor.process_record(raw_rec)
                        repo_proc_records.append(proc_rec)

                        # Accumulate metrics
                        repo_raw_lens.append(proc_rec.character_count_original)
                        repo_proc_lens.append(proc_rec.character_count_processed)
                        repo_tok_counts.append(proc_rec.token_count)
                        global_raw_lengths.append(proc_rec.character_count_original)
                        global_proc_lengths.append(proc_rec.character_count_processed)
                        global_token_counts.append(proc_rec.token_count)

                        if proc_rec.is_empty_after_preprocessing:
                            repo_empty_count += 1
                            total_empty_after_proc += 1
                        if proc_rec.has_code_block or proc_rec.has_inline_code:
                            repo_with_code += 1
                            total_comments_with_code += 1
                        if proc_rec.has_url:
                            repo_with_url += 1
                            total_comments_with_url += 1
                        if proc_rec.has_mention:
                            repo_with_mention += 1
                            total_comments_with_mention += 1
                        if proc_rec.has_issue_ref:
                            repo_with_issue += 1
                            total_comments_with_issue += 1

                        for lemma in proc_rec.lemmas:
                            repo_vocab[lemma] += 1
                            global_vocab[lemma] += 1

                        login = proc_rec.commenter_login
                        if login and login != "unknown":
                            repo_reviewers.add(login)
                            global_reviewers.add(login)
                            reviewer_comments[login] += 1
                            reviewer_tokens[login].append(proc_rec.token_count)

            # Write processed records
            with open(proc_path, "w", encoding="utf-8") as wf:
                for r in repo_proc_records:
                    wf.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

            total_processed_records += len(repo_proc_records)
            processed_files[full_name] = str(proc_path.resolve())

            repo_diagnostics[full_name] = {
                "repository": full_name,
                "raw_comments_count": repo_raw_count,
                "processed_comments_count": len(repo_proc_records),
                "empty_after_preprocessing": repo_empty_count,
                "empty_ratio": round(repo_empty_count / repo_raw_count, 4) if repo_raw_count else 0.0,
                "comments_with_code": repo_with_code,
                "comments_with_url": repo_with_url,
                "comments_with_mention": repo_with_mention,
                "comments_with_issue_ref": repo_with_issue,
                "unique_reviewers": len(repo_reviewers),
                "vocabulary_size": len(repo_vocab),
                "total_tokens": sum(repo_vocab.values()),
                "mean_token_count": round(float(np.mean(repo_tok_counts)), 2) if repo_tok_counts else 0.0,
                "median_token_count": round(float(np.median(repo_tok_counts)), 2) if repo_tok_counts else 0.0,
                "mean_raw_char_len": round(float(np.mean(repo_raw_lens)), 2) if repo_raw_lens else 0.0,
                "mean_proc_char_len": round(float(np.mean(repo_proc_lens)), 2) if repo_proc_lens else 0.0,
            }

            logger.info(
                f"Finished {full_name}: raw={repo_raw_count}, processed={len(repo_proc_records)}, "
                f"empty={repo_empty_count}, vocab={len(repo_vocab)}"
            )

        finish_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Vocabulary Analysis
        total_tokens_count = sum(global_vocab.values())
        unique_tokens_count = len(global_vocab)
        single_occurrence_tokens = sum(1 for tok, c in global_vocab.items() if c == 1)
        top_50_tokens = global_vocab.most_common(50)

        # Reviewer Statistics
        reviewer_stats = {}
        for login in sorted(global_reviewers):
            t_counts = reviewer_tokens.get(login, [])
            reviewer_stats[login] = {
                "comment_count": reviewer_comments.get(login, 0),
                "total_tokens": sum(t_counts),
                "mean_tokens_per_comment": round(float(np.mean(t_counts)), 2) if t_counts else 0.0,
                "median_tokens_per_comment": round(float(np.median(t_counts)), 2) if t_counts else 0.0,
            }

        # Build Preprocessing Report
        report = {
            "phase": "Phase 4 - NLP Preprocessing & Dataset Preparation",
            "timestamp": finish_time,
            "system_environment": {
                "python_version": platform.python_version(),
                "os": platform.system(),
                "nltk_version": getattr(nltk, "__version__", "unknown") if NLTK_AVAILABLE else "none",
                "lemmatizer": self.preprocessor.lemmatizer_name,
                "stopwords_source": self.preprocessor.stopwords_source,
            },
            "global_summary": {
                "total_repositories": len(target_repos),
                "total_raw_comments": total_raw_comments,
                "total_processed_records": total_processed_records,
                "empty_after_preprocessing": total_empty_after_proc,
                "empty_ratio": round(total_empty_after_proc / total_raw_comments, 4) if total_raw_comments else 0.0,
                "total_comments_with_code": total_comments_with_code,
                "total_comments_with_url": total_comments_with_url,
                "total_comments_with_mention": total_comments_with_mention,
                "total_comments_with_issue_ref": total_comments_with_issue,
                "total_unique_reviewers": len(global_reviewers),
                "mean_tokens_per_comment": round(float(np.mean(global_token_counts)), 2) if global_token_counts else 0.0,
                "median_tokens_per_comment": round(float(np.median(global_token_counts)), 2) if global_token_counts else 0.0,
                "mean_raw_char_length": round(float(np.mean(global_raw_lengths)), 2) if global_raw_lengths else 0.0,
                "median_raw_char_length": round(float(np.median(global_raw_lengths)), 2) if global_raw_lengths else 0.0,
                "mean_proc_char_length": round(float(np.mean(global_proc_lengths)), 2) if global_proc_lengths else 0.0,
                "median_proc_char_length": round(float(np.median(global_proc_lengths)), 2) if global_proc_lengths else 0.0,
            },
            "vocabulary_analysis": {
                "total_token_occurrences": total_tokens_count,
                "unique_token_count": unique_tokens_count,
                "hapax_legomena_count": single_occurrence_tokens,
                "hapax_legomena_ratio": round(single_occurrence_tokens / unique_tokens_count, 4) if unique_tokens_count else 0.0,
                "top_50_tokens": [{"token": tok, "frequency": count} for tok, count in top_50_tokens],
            },
            "per_repository_diagnostics": repo_diagnostics,
            "reviewer_summary": {
                "unique_reviewer_count": len(global_reviewers),
                "mean_comments_per_reviewer": round(total_raw_comments / len(global_reviewers), 2) if global_reviewers else 0.0,
                "reviewers": reviewer_stats,
            },
            "raw_file_verification": raw_checksums,
        }

        # Build Preprocessing Manifest
        manifest = {
            "dataset_name": "GDERS-10-Processed-Corpus",
            "version": "1.0.0",
            "phase": "Phase 4 - Preprocessed",
            "generated_at": finish_time,
            "source_manifest": str(manifest_file) if manifest_file else None,
            "collection_window": raw_manifest.get("collection_window", {}),
            "summary_statistics": {
                "total_repositories": len(target_repos),
                "raw_comments_count": total_raw_comments,
                "processed_comments_count": total_processed_records,
                "unique_reviewers": len(global_reviewers),
                "unique_vocabulary_size": unique_tokens_count,
                "total_tokens": total_tokens_count,
            },
            "nlp_pipeline_configuration": {
                "url_normalization": "stripped",
                "fenced_code_blocks": "stripped",
                "inline_code": "extracted_unquoted",
                "mentions_and_issues": "stripped",
                "special_characters": "stripped_with_hyphen_preservation",
                "lowercasing": True,
                "stopword_list": self.preprocessor.stopwords_source,
                "lemmatizer": self.preprocessor.lemmatizer_name,
            },
            "processed_files": processed_files,
            "raw_checksums": raw_checksums,
        }

        # Save manifest and report
        if self.config.preprocessing_manifest_file:
            with open(self.config.preprocessing_manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Saved preprocessing manifest to {self.config.preprocessing_manifest_file}")

        if self.config.preprocessing_report_file:
            with open(self.config.preprocessing_report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved preprocessing report to {self.config.preprocessing_report_file}")

        return report

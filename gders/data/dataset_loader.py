"""
GDERS Dataset Loader Module.

Handles transparent reading and writing of raw review comments, processed tokens,
and developer-to-comment associations in simple, portable formats (JSON / JSONL).
"""

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from gders.config import DEFAULT_CONFIG, GDERSConfig
from gders.data.repo_collector import ReviewCommentRecord


class GDERSDatasetLoader:
    """Loader and manager for GDERS raw and processed dataset artifacts."""

    def __init__(self, config: Optional[GDERSConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.config.ensure_directories()

    def save_raw_comments(
        self,
        owner: str,
        repo: str,
        comments: List[ReviewCommentRecord],
        append: bool = True,
    ) -> Path:
        """Save a batch of review comment records to data/gders/raw_comments/{owner}__{repo}.jsonl."""
        file_path = self.config.raw_comments_dir / f"{owner}__{repo}.jsonl"
        mode = "a" if append else "w"

        with open(file_path, mode, encoding="utf-8") as f:
            for record in comments:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return file_path

    def load_raw_comments_for_repo(
        self,
        owner: str,
        repo: str,
    ) -> List[ReviewCommentRecord]:
        """Load all review comments for a specific target repository."""
        file_path = self.config.raw_comments_dir / f"{owner}__{repo}.jsonl"
        if not file_path.exists():
            return []

        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                records.append(ReviewCommentRecord(**d))
        return records

    def iter_all_raw_comments(self) -> Generator[ReviewCommentRecord, None, None]:
        """Iterate over all collected raw review comments across all target repositories."""
        for file_path in sorted(self.config.raw_comments_dir.glob("*.jsonl")):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    yield ReviewCommentRecord(**d)

    def save_processed_comments(
        self,
        owner: str,
        repo: str,
        comments: List[Any],
        append: bool = False,
    ) -> Path:
        """Save a batch of processed comment records to data/gders/processed/comments/{owner}__{repo}.jsonl."""
        self.config.processed_comments_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.config.processed_comments_dir / f"{owner}__{repo}.jsonl"
        mode = "a" if append else "w"

        with open(file_path, mode, encoding="utf-8") as f:
            for record in comments:
                d = record.to_dict() if hasattr(record, "to_dict") else record
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return file_path

    def load_processed_comments_for_repo(
        self,
        owner: str,
        repo: str,
    ) -> List[Dict[str, Any]]:
        """Load all processed review comments for a specific target repository."""
        file_path = self.config.processed_comments_dir / f"{owner}__{repo}.jsonl"
        if not file_path.exists():
            return []

        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                records.append(d)
        return records

    def iter_all_processed_comments(self) -> Generator[Dict[str, Any], None, None]:
        """Iterate over all preprocessed review comments across all target repositories."""
        if not self.config.processed_comments_dir.exists():
            return
        for file_path in sorted(self.config.processed_comments_dir.glob("*.jsonl")):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    yield d

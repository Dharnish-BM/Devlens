"""
GDERS Repository & PR Review Comment Collector.

Provides clean, structured abstractions for fetching repository metadata,
pull requests, and pull request review comments with pagination, rate-limit
awareness, and resumable state tracking.

Reuses the safe, generic GitHubClient from devlens.data_collection.github_client
without modifying it or coupling to DevLens profile pipelines.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set

from devlens.data_collection.github_client import GitHubClient
from gders.config import DEFAULT_CONFIG, GDERSConfig, TargetRepository

logger = logging.getLogger(__name__)


@dataclass
class ReviewCommentRecord:
    """Standardized GDERS PR Review Comment Data Contract with Complete Provenance."""
    repository: str
    owner: str
    repo: str
    repository_url: str
    pull_request_number: int
    pull_request_url: str
    pull_request_title: str
    pull_request_author: str
    comment_id: int
    commenter_login: str
    commenter_id: Optional[int]
    comment_body: str
    created_at: str
    updated_at: str
    comment_url: str
    commit_id: Optional[str] = None
    file_path: Optional[str] = None
    diff_hunk: Optional[str] = None
    collection_timestamp: Optional[str] = None
    collection_window_start: Optional[str] = None
    collection_window_end: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PullRequestSummary:
    """Pull Request metadata container for GDERS selection and linking."""
    repository: str
    number: int
    title: str
    author_login: str
    created_at: str
    updated_at: str
    closed_at: Optional[str]
    merged_at: Optional[str]
    state: str
    html_url: str
    body: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    changed_files: Optional[int] = None
    additions: Optional[int] = None
    deletions: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GDERSRepoCollector:
    """Collector for target open source repositories in GDERS."""

    def __init__(
        self,
        config: Optional[GDERSConfig] = None,
        client: Optional[GitHubClient] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.client = client or GitHubClient()
        self.config.ensure_directories()
        self._file_handler: Optional[logging.Handler] = None
        self._setup_file_logging()

    def _setup_file_logging(self) -> None:
        """Setup GDERS collection file logger."""
        if not self.config.collection_log_file:
            return
        try:
            self.config.collection_log_file.parent.mkdir(parents=True, exist_ok=True)
            log_path_str = str(self.config.collection_log_file.resolve())
            file_handler_exists = any(
                isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path_str
                for h in logger.handlers
            )
            if not file_handler_exists:
                fh = logging.FileHandler(str(self.config.collection_log_file), encoding="utf-8")
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] [GDERS_COLLECTOR] %(message)s")
                fh.setFormatter(formatter)
                logger.addHandler(fh)
                self._file_handler = fh
        except Exception:
            pass

    def close(self) -> None:
        """Release any open file handlers."""
        if self._file_handler:
            try:
                self._file_handler.close()
                logger.removeHandler(self._file_handler)
                self._file_handler = None
            except Exception:
                pass

    def fetch_repo_metadata(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Fetch general repository metadata (stars, forks, description, primary language)."""
        return self.client.rest_request(f"repos/{owner}/{repo}")

    def iter_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        max_prs: Optional[int] = None,
        cutoff_date: Optional[str] = None,
    ) -> Generator[PullRequestSummary, None, None]:
        """Paginate and yield pull request summaries for a target repository."""
        limit = max_prs or self.config.max_prs_per_repo
        page = 1
        yielded = 0

        while yielded < limit:
            per_page = min(self.config.per_page_pagination, limit - yielded)
            pulls = self.client.rest_request(
                f"repos/{owner}/{repo}/pulls",
                params={"state": state, "per_page": per_page, "page": page, "sort": "updated", "direction": "desc"},
            )
            if not pulls or not isinstance(pulls, list):
                break

            for p in pulls:
                if not isinstance(p, dict):
                    continue
                created_at = p.get("created_at") or ""
                if cutoff_date and created_at < cutoff_date:
                    return

                author = p.get("user", {}).get("login") if isinstance(p.get("user"), dict) else "unknown"
                pr_summary = PullRequestSummary(
                    repository=f"{owner}/{repo}",
                    number=p.get("number", 0),
                    title=p.get("title", ""),
                    author_login=author,
                    created_at=created_at,
                    updated_at=p.get("updated_at") or "",
                    closed_at=p.get("closed_at"),
                    merged_at=p.get("merged_at"),
                    state=p.get("state", "open"),
                    html_url=p.get("html_url", ""),
                    body=p.get("body"),
                    labels=[lbl.get("name", "") for lbl in p.get("labels", []) if isinstance(lbl, dict)],
                )
                yield pr_summary
                yielded += 1
                if yielded >= limit:
                    break

            if len(pulls) < per_page:
                break
            page += 1

    def fetch_pr_review_comments(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        pull_request_url: str = "",
        pull_request_title: str = "",
        pull_request_author: str = "",
        window_start: Optional[str] = None,
        window_end: Optional[str] = None,
    ) -> List[ReviewCommentRecord]:
        """Fetch code review comments on a specific pull request with complete provenance."""
        endpoint = f"repos/{owner}/{repo}/pulls/{pull_number}/comments"
        raw_comments = self.client.rest_request(endpoint, params={"per_page": self.config.max_comments_per_pr})
        if not raw_comments or not isinstance(raw_comments, list):
            return []

        collection_now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        records: List[ReviewCommentRecord] = []
        for c in raw_comments:
            if not isinstance(c, dict):
                continue
            user_obj = c.get("user") or {}
            login = user_obj.get("login") if isinstance(user_obj, dict) else "unknown"
            uid = user_obj.get("id") if isinstance(user_obj, dict) else None

            record = ReviewCommentRecord(
                repository=f"{owner}/{repo}",
                owner=owner,
                repo=repo,
                repository_url=f"https://github.com/{owner}/{repo}",
                pull_request_number=pull_number,
                pull_request_url=pull_request_url or c.get("pull_request_url", f"https://github.com/{owner}/{repo}/pull/{pull_number}"),
                pull_request_title=pull_request_title,
                pull_request_author=pull_request_author,
                comment_id=c.get("id", 0),
                commenter_login=login,
                commenter_id=uid,
                comment_body=c.get("body", ""),
                created_at=c.get("created_at", ""),
                updated_at=c.get("updated_at", ""),
                comment_url=c.get("html_url", ""),
                commit_id=c.get("commit_id"),
                file_path=c.get("path"),
                diff_hunk=c.get("diff_hunk"),
                collection_timestamp=collection_now,
                collection_window_start=window_start,
                collection_window_end=window_end,
            )
            records.append(record)
        return records

    def get_repo_raw_storage_path(self, owner: str, repo: str) -> Path:
        """Return dedicated path for storing a repository's review comments."""
        return self.config.raw_comments_dir / f"{owner}__{repo}.jsonl"

    def load_collection_state(self) -> Dict[str, Any]:
        """Load persistent collection state from collection_state.json."""
        state_file = self.config.collection_state_file
        if state_file and state_file.exists():
            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not read collection state: {e}")
        return {"repositories": {}, "last_updated": None}

    def save_collection_state(self, state: Dict[str, Any]) -> None:
        """Save persistent collection state to collection_state.json."""
        state_file = self.config.collection_state_file
        if state_file:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state["last_updated"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

    def collect_repository(
        self,
        owner: str,
        repo: str,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Collect PR review comments for a single repository with safety limits, deduplication, and resumability.
        
        Saves raw comments in data/gders/raw_comments/{owner}__{repo}.jsonl
        """
        full_name = f"{owner}/{repo}"
        logger.info(f"Starting GDERS collection for {full_name}")
        start_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Determine collection window
        if self.config.collection_end_date:
            end_dt = datetime.strptime(self.config.collection_end_date[:19], "%Y-%m-%dT%H:%M:%S")
        else:
            end_dt = datetime.utcnow()
        if self.config.collection_start_date:
            start_dt = datetime.strptime(self.config.collection_start_date[:19], "%Y-%m-%dT%H:%M:%S")
        else:
            start_dt = end_dt - timedelta(days=self.config.lookback_days)

        window_start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        window_end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Load existing collection state
        state_dict = self.load_collection_state()
        repo_state = state_dict.get("repositories", {}).get(full_name, {})

        if resume and repo_state.get("status") == "completed":
            logger.info(f"Repository {full_name} is already completed in collection state. Skipping.")
            return repo_state

        # Storage file & existing comment IDs for deduplication
        raw_file = self.get_repo_raw_storage_path(owner, repo)
        existing_comment_ids: Set[int] = set()
        existing_pr_numbers: Set[int] = set()

        if raw_file.exists():
            with open(raw_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "comment_id" in rec:
                            existing_comment_ids.add(rec["comment_id"])
                        if "pull_request_number" in rec:
                            existing_pr_numbers.add(rec["pull_request_number"])
                    except Exception:
                        pass

        # Track collection metrics
        initial_api_calls = self.client.api_calls_count
        api_errors = 0
        rate_limit_events = 0
        prs_discovered = 0
        prs_collected = len(existing_pr_numbers)
        comments_collected = len(existing_comment_ids)
        unique_reviewers: Set[str] = set()

        # Update state to 'collecting'
        repo_state = {
            "repository": full_name,
            "status": "collecting",
            "collection_started_at": start_time,
            "collection_finished_at": None,
            "collection_window_start": window_start_str,
            "collection_window_end": window_end_str,
            "prs_discovered": prs_discovered,
            "prs_collected": prs_collected,
            "review_comments_discovered": comments_collected,
            "review_comments_collected": comments_collected,
            "unique_reviewers": 0,
            "api_requests": 0,
            "api_errors": api_errors,
            "rate_limit_events": rate_limit_events,
            "file_path": str(raw_file),
            "safety_limit_hit": None,
        }
        state_dict.setdefault("repositories", {})[full_name] = repo_state
        self.save_collection_state(state_dict)

        # Open file in append mode
        status = "completed"
        safety_limit_hit = None

        try:
            with open(raw_file, "a", encoding="utf-8") as out_f:
                for pr in self.iter_pull_requests(
                    owner,
                    repo,
                    state="all",
                    max_prs=self.config.max_prs_per_repo,
                    cutoff_date=window_start_str,
                ):
                    prs_discovered += 1

                    # Check max API requests safety limit
                    current_api_calls = self.client.api_calls_count - initial_api_calls
                    if current_api_calls >= self.config.max_api_requests_per_repo:
                        safety_limit_hit = f"MAX_API_REQUESTS_PER_REPOSITORY ({self.config.max_api_requests_per_repo})"
                        status = "partial"
                        logger.warning(f"Safety limit hit for {full_name}: {safety_limit_hit}")
                        break

                    # Check max comments safety limit
                    if comments_collected >= self.config.max_comments_per_repo:
                        safety_limit_hit = f"MAX_COMMENTS_PER_REPOSITORY ({self.config.max_comments_per_repo})"
                        status = "partial"
                        logger.warning(f"Safety limit hit for {full_name}: {safety_limit_hit}")
                        break

                    # Skip PR if already fully processed in previous run
                    if pr.number in existing_pr_numbers:
                        continue

                    # Fetch review comments for this PR
                    try:
                        comments = self.fetch_pr_review_comments(
                            owner=owner,
                            repo=repo,
                            pull_number=pr.number,
                            pull_request_url=pr.html_url,
                            pull_request_title=pr.title,
                            pull_request_author=pr.author_login,
                            window_start=window_start_str,
                            window_end=window_end_str,
                        )
                    except Exception as e:
                        api_errors += 1
                        logger.error(f"Error fetching comments for {full_name} PR #{pr.number}: {e}")
                        continue

                    # Deduplicate and write
                    new_comments_in_pr = 0
                    for c in comments:
                        if c.comment_id in existing_comment_ids:
                            continue
                        existing_comment_ids.add(c.comment_id)
                        if c.commenter_login and c.commenter_login != "unknown":
                            unique_reviewers.add(c.commenter_login)

                        out_f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                        comments_collected += 1
                        new_comments_in_pr += 1

                        if comments_collected >= self.config.max_comments_per_repo:
                            break

                    prs_collected += 1
                    existing_pr_numbers.add(pr.number)

                    if comments_collected >= self.config.max_comments_per_repo:
                        safety_limit_hit = f"MAX_COMMENTS_PER_REPOSITORY ({self.config.max_comments_per_repo})"
                        status = "partial"
                        break

        except Exception as e:
            status = "failed"
            api_errors += 1
            logger.error(f"Fatal collection error on {full_name}: {e}")

        finish_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        raw_api_count = getattr(self.client, "api_calls_count", 0)
        try:
            total_api_requests = int(raw_api_count) - int(initial_api_calls)
        except Exception:
            total_api_requests = 0

        # Update final repository state
        repo_state.update({
            "status": status,
            "collection_finished_at": finish_time,
            "prs_discovered": prs_discovered,
            "prs_collected": prs_collected,
            "review_comments_discovered": comments_collected,
            "review_comments_collected": comments_collected,
            "unique_reviewers": len(unique_reviewers),
            "api_requests": total_api_requests,
            "api_errors": api_errors,
            "rate_limit_events": rate_limit_events,
            "safety_limit_hit": safety_limit_hit,
        })
        state_dict["repositories"][full_name] = repo_state
        self.save_collection_state(state_dict)

        logger.info(
            f"Finished GDERS collection for {full_name}: status={status}, "
            f"prs={prs_collected}, comments={comments_collected}, reviewers={len(unique_reviewers)}"
        )
        return repo_state

    def collect_full_dataset(
        self,
        repositories: Optional[List[TargetRepository]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Collect PR review comments across all target repositories and build manifest."""
        targets = repositories or self.config.target_repositories
        results = {}

        logger.info(f"=== Starting GDERS Full Dataset Collection for {len(targets)} repositories ===")

        for target in targets:
            res = self.collect_repository(owner=target.owner, repo=target.name, resume=resume)
            results[target.full_name] = res

        manifest = self.generate_dataset_manifest(results)
        return manifest

    def generate_dataset_manifest(self, repo_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate and save data/gders/processed/dataset_manifest.json with dataset-wide provenance and stats."""
        state_dict = self.load_collection_state()
        repositories_state = state_dict.get("repositories", {})
        if repo_results:
            repositories_state.update(repo_results)

        total_prs = sum(r.get("prs_collected", 0) for r in repositories_state.values())
        total_comments = sum(r.get("review_comments_collected", 0) for r in repositories_state.values())
        total_api_requests = sum(r.get("api_requests", 0) for r in repositories_state.values())
        total_api_errors = sum(r.get("api_errors", 0) for r in repositories_state.values())
        total_rate_limits = sum(r.get("rate_limit_events", 0) for r in repositories_state.values())

        # Aggregate unique reviewers across files
        all_reviewers: Set[str] = set()
        repo_files = {}
        for r_name, r_data in repositories_state.items():
            f_path = r_data.get("file_path")
            if f_path and Path(f_path).exists():
                repo_files[r_name] = str(Path(f_path).resolve())
                with open(f_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            login = d.get("commenter_login")
                            if login and login != "unknown":
                                all_reviewers.add(login)
                        except Exception:
                            pass

        # Determine overall collection window
        window_start = self.config.collection_start_date
        window_end = self.config.collection_end_date
        if not window_end:
            window_end = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        if not window_start:
            end_dt = datetime.strptime(window_end[:19], "%Y-%m-%dT%H:%M:%S")
            window_start = (end_dt - timedelta(days=self.config.lookback_days)).strftime("%Y-%m-%dT%H:%M:%SZ")

        manifest = {
            "dataset_name": "GDERS-10-Corpus",
            "version": "1.0.0",
            "phase": "Phase 3B - Full Collection",
            "collection_window": {
                "collection_start_date": window_start,
                "collection_end_date": window_end,
                "lookback_days": self.config.lookback_days,
            },
            "collection_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary_statistics": {
                "total_repositories": len(repositories_state),
                "total_prs_collected": total_prs,
                "total_review_comments_collected": total_comments,
                "total_unique_reviewers": len(all_reviewers),
                "total_api_requests": total_api_requests,
                "total_api_errors": total_api_errors,
                "total_rate_limit_events": total_rate_limits,
            },
            "safety_limits": {
                "max_prs_per_repo": self.config.max_prs_per_repo,
                "max_comments_per_repo": self.config.max_comments_per_repo,
                "max_api_requests_per_repo": self.config.max_api_requests_per_repo,
            },
            "status_summary": {
                "completed": [k for k, v in repositories_state.items() if v.get("status") == "completed"],
                "partial": [k for k, v in repositories_state.items() if v.get("status") == "partial"],
                "failed": [k for k, v in repositories_state.items() if v.get("status") == "failed"],
            },
            "repository_files": repo_files,
            "repositories": repositories_state,
        }

        # Write manifest file
        if self.config.dataset_manifest_file:
            self.config.dataset_manifest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.dataset_manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Dataset manifest saved to {self.config.dataset_manifest_file}")

        return manifest

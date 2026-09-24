"""
GDERS Repository Suitability Assessment & Controlled Pilot Module.

Conducts low-cost, rate-limited suitability inspection of target open-source repositories:
- Fetches repository metadata
- Collects controlled sample of recent Pull Requests within a pilot window
- Extracts PR review comments specifically (distinct from issue/general discussion comments)
- Applies transparent "pilot technical-comment heuristic"
- Evaluates reviewer density, comment length statistics, and technical signal ratio
- Produces machine-readable JSON and human-readable Markdown suitability reports
- Categorizes candidates as Strong, Usable, Weak, or Infeasible
"""

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from devlens.data_collection.github_client import GitHubClient
from gders.config import DEFAULT_CONFIG, GDERSConfig, TargetRepository
from gders.data.repo_collector import GDERSRepoCollector, PullRequestSummary, ReviewCommentRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Technical Comment Quality Heuristics (Transparent Phase 2 Pilot Rules)
# ---------------------------------------------------------------------------
TECHNICAL_KEYWORD_PATTERNS = [
    r"\b(api|class|function|method|interface|struct|trait|module|package|endpoint)\b",
    r"\b(refactor|optimize|performance|memory|leak|alloc|concurrency|thread|async|await)\b",
    r"\b(test|spec|mock|fixture|assertion|coverage|ci|pipeline|workflow)\b",
    r"\b(security|vulnerability|sanitize|escape|auth|token|permission|csrf|xss)\b",
    r"\b(bug|fix|error|exception|crash|null|undefined|race|deadlock|timeout)\b",
    r"\b(param|argument|type|return|schema|validate|deprecated|override)\b",
    r"\b(architecture|pattern|decouple|dependency|injection|singleton|factory)\b",
    r"\b(database|sql|query|index|cache|redis|postgres|migration)\b",
]

NOISE_PATTERNS = [
    r"^(lgtm|looks good to me|looks good|sounds good|thanks|thank you|\+1|approved|done|ack|nit|ok|okay)[\.\!]?$",
    r"^[\U00010000-\U0010ffff\s\:\+\-]+$",  # pure emoji / reaction symbols
]


def is_technical_comment_heuristic(
    body: str,
    min_length: int = 15,
    min_word_count: int = 3,
) -> Tuple[bool, str]:
    """Transparent heuristic to separate technically meaningful PR review comments from conversational noise.
    
    Returns:
        (is_technical: bool, reason: str)
    """
    if not body or not isinstance(body, str):
        return False, "empty_body"

    text = body.strip()
    if len(text) < min_length:
        return False, f"too_short (<{min_length} chars)"

    words = text.split()
    if len(words) < min_word_count:
        return False, f"too_few_words (<{min_word_count} words)"

    text_lower = text.lower()

    # Noise rejection
    for n_pat in NOISE_PATTERNS:
        if re.match(n_pat, text_lower):
            return False, "noise_acknowledgement"

    # Bot / automated message filtering
    if "automatically generated" in text_lower or "dependabot" in text_lower or "cla-assistant" in text_lower:
        return False, "automated_bot_comment"

    # Code presence boost (inline code `foo` or code blocks ```)
    has_code = "`" in text or "```" in text

    # Technical keyword pattern check
    matched_patterns = []
    for pat in TECHNICAL_KEYWORD_PATTERNS:
        if re.search(pat, text_lower):
            matched_patterns.append(pat)

    if has_code or len(matched_patterns) >= 1:
        return True, f"technical_signal (code_snippet={has_code}, keywords={len(matched_patterns)})"

    # If longer than 40 chars and has natural review language (e.g. "should", "could", "why", "we need")
    if len(text) >= 40 and any(kw in text_lower for kw in ("should", "instead", "consider", "better", "why", "need to", "make sure")):
        return True, "review_suggestion_structure"

    return False, "low_technical_signal"


@dataclass
class RepoPilotMetrics:
    """Suitability metrics for an individual evaluated repository."""
    repository_name: str
    owner: str
    repo: str
    primary_language: Optional[str] = None
    description: Optional[str] = None
    stars: Optional[int] = None
    forks: Optional[int] = None
    open_issues: Optional[int] = None
    
    collection_window_start: str = ""
    collection_window_end: str = ""
    
    sampled_prs: int = 0
    sampled_prs_with_review_comments: int = 0
    sampled_review_comments: int = 0
    sampled_unique_reviewers: int = 0
    sampled_technical_comments: int = 0
    
    avg_comment_length: float = 0.0
    median_comment_length: float = 0.0
    short_comment_ratio: float = 0.0
    technical_signal_ratio: float = 0.0
    
    api_errors: int = 0
    rate_limit_events: int = 0
    api_status: str = "OK"
    last_activity_date: Optional[str] = None
    
    suitability_category: str = "Usable candidate"  # Strong, Usable, Weak, Infeasible
    suitability_rationale: str = ""
    domain_classification: str = "General"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RepositoryAssessmentRunner:
    """Orchestrates lightweight repository suitability inspection and controlled pilot."""

    def __init__(
        self,
        config: Optional[GDERSConfig] = None,
        client: Optional[GitHubClient] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.client = client or GitHubClient()
        self.collector = GDERSRepoCollector(config=self.config, client=self.client)
        self.config.ensure_directories()

    def determine_collection_window(self) -> Tuple[str, str]:
        """Compute ISO start and end timestamps for the evaluation window."""
        if self.config.collection_start_date and self.config.collection_end_date:
            return self.config.collection_start_date, self.config.collection_end_date

        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=self.config.lookback_days)
        return start_dt.isoformat() + "Z", end_dt.isoformat() + "Z"

    def evaluate_single_repository(
        self,
        target: TargetRepository,
        max_prs: Optional[int] = None,
    ) -> Tuple[RepoPilotMetrics, List[ReviewCommentRecord]]:
        """Evaluate a single repository with controlled sampling."""
        owner = target.owner
        repo = target.name
        full_name = target.full_name
        limit = max_prs or self.config.pilot_max_prs_per_repo
        start_win, end_win = self.determine_collection_window()

        logger.info(f"[GDERS Pilot] Assessing repository: '{full_name}' (max_prs={limit})...")

        meta = self.collector.fetch_repo_metadata(owner, repo) or {}
        api_errors = 0
        api_status = "OK"

        if not meta:
            api_errors += 1
            api_status = "METADATA_FETCH_FAILED"
            logger.warning(f"[GDERS Pilot] Metadata fetch failed for '{full_name}'.")

        stars = meta.get("stargazers_count")
        forks = meta.get("forks_count")
        open_issues = meta.get("open_issues_count")
        lang = meta.get("language") or target.primary_language
        desc = meta.get("description") or target.description
        last_pushed = meta.get("pushed_at")

        # Sample PRs
        sampled_prs: List[PullRequestSummary] = []
        try:
            for pr in self.collector.iter_pull_requests(
                owner=owner,
                repo=repo,
                state="all",
                max_prs=limit,
                cutoff_date=start_win,
            ):
                sampled_prs.append(pr)
        except Exception as e:
            api_errors += 1
            api_status = f"PR_ITERATION_ERROR: {str(e)}"
            logger.error(f"[GDERS Pilot] Error sampling PRs for '{full_name}': {e}")

        # Fetch comments for sampled PRs
        all_comments: List[ReviewCommentRecord] = []
        prs_with_comments = 0
        unique_reviewers: Set[str] = set()
        seen_comment_ids: Set[int] = set()

        for pr in sampled_prs:
            try:
                pr_comments = self.collector.fetch_pr_review_comments(
                    owner=owner,
                    repo=repo,
                    pull_number=pr.number,
                    pull_request_url=pr.html_url,
                )
                if pr_comments:
                    prs_with_comments += 1
                    for c in pr_comments:
                        # Deduplicate comment IDs
                        if c.comment_id not in seen_comment_ids:
                            seen_comment_ids.add(c.comment_id)
                            all_comments.append(c)
                            if c.commenter_login and c.commenter_login != "unknown":
                                unique_reviewers.add(c.commenter_login)
            except Exception as e:
                api_errors += 1
                logger.warning(f"[GDERS Pilot] Error fetching comments for {full_name}#{pr.number}: {e}")

        # Compute comment length & technical heuristics
        comment_lengths = [len(c.comment_body) for c in all_comments]
        avg_len = float(np.mean(comment_lengths)) if comment_lengths else 0.0
        med_len = float(np.median(comment_lengths)) if comment_lengths else 0.0
        short_count = sum(1 for l in comment_lengths if l < 25)
        short_ratio = (short_count / max(len(all_comments), 1)) if all_comments else 0.0

        technical_comments = 0
        for c in all_comments:
            is_tech, _ = is_technical_comment_heuristic(c.comment_body)
            if is_tech:
                technical_comments += 1

        tech_signal_ratio = (technical_comments / max(len(all_comments), 1)) if all_comments else 0.0

        # Domain classification heuristic
        domain = self._classify_domain(full_name, lang, desc or "")

        # Suitability classification
        suitability_cat, rationale = self._classify_suitability(
            prs_count=len(sampled_prs),
            prs_with_comments=prs_with_comments,
            total_comments=len(all_comments),
            unique_reviewers=len(unique_reviewers),
            tech_ratio=tech_signal_ratio,
            api_status=api_status,
        )

        metrics = RepoPilotMetrics(
            repository_name=full_name,
            owner=owner,
            repo=repo,
            primary_language=lang,
            description=desc,
            stars=stars,
            forks=forks,
            open_issues=open_issues,
            collection_window_start=start_win,
            collection_window_end=end_win,
            sampled_prs=len(sampled_prs),
            sampled_prs_with_review_comments=prs_with_comments,
            sampled_review_comments=len(all_comments),
            sampled_unique_reviewers=len(unique_reviewers),
            sampled_technical_comments=technical_comments,
            avg_comment_length=round(avg_len, 1),
            median_comment_length=round(med_len, 1),
            short_comment_ratio=round(short_ratio, 3),
            technical_signal_ratio=round(tech_signal_ratio, 3),
            api_errors=api_errors,
            rate_limit_events=0,
            api_status=api_status,
            last_activity_date=last_pushed,
            suitability_category=suitability_cat,
            suitability_rationale=rationale,
            domain_classification=domain,
        )

        return metrics, all_comments

    def run_assessment(
        self,
        repositories: Optional[List[TargetRepository]] = None,
        max_prs_per_repo: Optional[int] = None,
        output_prefix: str = "repository_suitability",
        report_title: str = "GDERS Repository Suitability Assessment & Controlled Pilot Report",
    ) -> Dict[str, Any]:
        """Run suitability assessment across all configured repositories."""
        target_list = repositories or self.config.target_repositories
        limit = max_prs_per_repo or self.config.pilot_max_prs_per_repo

        all_metrics: List[RepoPilotMetrics] = []
        pilot_data_by_repo: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(f"[GDERS Pilot] Starting assessment across {len(target_list)} repositories...")

        for target in target_list:
            metrics, comments = self.evaluate_single_repository(target, max_prs=limit)
            all_metrics.append(metrics)

            # Persist pilot raw comments in dedicated data/gders/raw_comments/pilot/{owner}__{repo}.jsonl
            pilot_file = self.config.pilot_raw_dir / f"{target.owner}__{target.name}.jsonl"
            with open(pilot_file, "w", encoding="utf-8") as pf:
                for c in comments:
                    pf.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
            
            pilot_data_by_repo[target.full_name] = [c.to_dict() for c in comments]

        # Generate output reports
        report_data = self._generate_reports(all_metrics, output_prefix=output_prefix, report_title=report_title)
        return report_data

    def _classify_domain(self, repo_name: str, language: Optional[str], desc: str) -> str:
        """Categorize repository into broad software engineering domain."""
        n = repo_name.lower()
        d = desc.lower()
        if "react" in n or "frontend" in d:
            return "Frontend / UI Library"
        if "rails" in n or "django" in d or "spring" in n or "flask" in n:
            return "Backend Web Framework"
        if "node" in n or "runtime" in d:
            return "JavaScript Runtime & Systems"
        if "kubernetes" in n or "cloud" in d or "container" in d:
            return "Cloud & Infrastructure"
        if "flutter" in n or "mobile" in d:
            return "Mobile Framework"
        if "tensorflow" in n or "scikit-learn" in n or "machine learning" in d:
            return "Data Science & Machine Learning"
        if "vscode" in n or "editor" in d:
            return "Developer Tooling & Desktop"
        if "spark" in n or "distributed" in d:
            return "Distributed Data Processing"
        if "elasticsearch" in n or "search" in d or "kafka" in n or "streaming" in d:
            return "Distributed Search & Event Streaming"
        return "Software Library"

    def _classify_suitability(
        self,
        prs_count: int,
        prs_with_comments: int,
        total_comments: int,
        unique_reviewers: int,
        tech_ratio: float,
        api_status: str,
    ) -> Tuple[str, str]:
        """Classify candidate into Strong, Usable, Weak, or Infeasible based on transparent metrics."""
        if api_status != "OK" or prs_count == 0:
            return "Infeasible", "API connectivity failed or zero pull requests accessible in collection window."

        if total_comments >= 25 and unique_reviewers >= 8 and tech_ratio >= 0.50:
            return "Strong candidate", f"High review comment density ({total_comments} comments across {unique_reviewers} reviewers) with strong technical content ({tech_ratio:.1%})."

        if total_comments >= 10 and unique_reviewers >= 3 and tech_ratio >= 0.40:
            return "Usable candidate", f"Moderate comment density ({total_comments} comments across {unique_reviewers} reviewers, {tech_ratio:.1%} technical signal)."

        if total_comments < 10 or unique_reviewers < 3:
            return "Weak candidate", f"Low review participation in sample ({total_comments} comments, {unique_reviewers} reviewers). May require wider historical window."

        return "Usable candidate", "Meets basic volume and quality criteria."

    def _generate_reports(
        self,
        metrics_list: List[RepoPilotMetrics],
        output_prefix: str = "repository_suitability",
        report_title: str = "GDERS Repository Suitability Assessment & Controlled Pilot Report",
    ) -> Dict[str, Any]:
        """Save JSON and Markdown suitability reports."""
        start_win, end_win = self.determine_collection_window()
        total_sampled_prs = sum(m.sampled_prs for m in metrics_list)
        total_comments = sum(m.sampled_review_comments for m in metrics_list)
        total_reviewers = sum(m.sampled_unique_reviewers for m in metrics_list)
        total_tech_comments = sum(m.sampled_technical_comments for m in metrics_list)

        report_summary = {
            "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
            "pilot_window": {
                "start": start_win,
                "end": end_win,
                "lookback_days": self.config.lookback_days,
            },
            "aggregate_statistics": {
                "repositories_evaluated": len(metrics_list),
                "total_prs_sampled": total_sampled_prs,
                "total_review_comments_collected": total_comments,
                "total_unique_reviewers_observed": total_reviewers,
                "total_technical_comments": total_tech_comments,
                "overall_technical_signal_ratio": round(total_tech_comments / max(total_comments, 1), 3),
            },
            "repositories": [m.to_dict() for m in metrics_list],
        }

        # Save JSON report to data/gders/processed/pilot/{output_prefix}.json
        json_path = self.config.pilot_processed_dir / f"{output_prefix}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(report_summary, jf, indent=2, ensure_ascii=False)

        # Save Markdown report to data/gders/processed/pilot/{output_prefix}.md
        md_path = self.config.pilot_processed_dir / f"{output_prefix}.md"
        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write(self._build_markdown_report(report_summary, metrics_list, title=report_title))

        logger.info(f"[GDERS Pilot] Reports saved to '{json_path}' and '{md_path}'.")
        return report_summary

    def _build_markdown_report(
        self,
        summary: Dict[str, Any],
        metrics_list: List[RepoPilotMetrics],
        title: str = "GDERS Repository Suitability Assessment & Controlled Pilot Report",
    ) -> str:
        """Construct the Markdown suitability table and narrative."""
        lines = [
            f"# {title}",
            "",
            f"- **Evaluation Date**: `{summary['evaluation_timestamp']}`",
            f"- **Pilot Window**: `{summary['pilot_window']['start']}` to `{summary['pilot_window']['end']}` ({summary['pilot_window']['lookback_days']} days)",
            f"- **Repositories Evaluated**: {summary['aggregate_statistics']['repositories_evaluated']}",
            f"- **Total Sampled PRs**: {summary['aggregate_statistics']['total_prs_sampled']}",
            f"- **Total Review Comments**: {summary['aggregate_statistics']['total_review_comments_collected']}",
            f"- **Total Unique Reviewers**: {summary['aggregate_statistics']['total_unique_reviewers_observed']}",
            f"- **Overall Technical Signal Ratio**: {summary['aggregate_statistics']['overall_technical_signal_ratio']:.1%}",
            "",
            "## Repository Suitability Summary Table",
            "",
            "| Repository | Domain | Language | PRs Sampled | PRs With Comments | Review Comments | Unique Reviewers | Technical Comments | Avg Length | Suitability |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]

        for m in metrics_list:
            lines.append(
                f"| `{m.repository_name}` | {m.domain_classification} | {m.primary_language or 'N/A'} | "
                f"{m.sampled_prs} | {m.sampled_prs_with_review_comments} | {m.sampled_review_comments} | "
                f"{m.sampled_unique_reviewers} | {m.sampled_technical_comments} | {m.avg_comment_length} chars | "
                f"**{m.suitability_category}** |"
            )

        lines.extend([
            "",
            "## Repository-by-Repository Assessment",
            "",
        ])

        for m in metrics_list:
            lines.extend([
                f"### {m.repository_name}",
                f"- **Primary Language**: {m.primary_language or 'Unknown'}",
                f"- **Domain Area**: {m.domain_classification}",
                f"- **Stars**: {m.stars or 'N/A'} | **Forks**: {m.forks or 'N/A'} | **Open Issues**: {m.open_issues or 'N/A'}",
                f"- **Review Comments**: {m.sampled_review_comments} across {m.sampled_unique_reviewers} unique reviewers (in {m.sampled_prs} sampled PRs)",
                f"- **Technical Quality**: {m.sampled_technical_comments}/{m.sampled_review_comments} ({m.technical_signal_ratio:.1%}) technical comments, average length {m.avg_comment_length} chars (median: {m.median_comment_length} chars)",
                f"- **Suitability Rating**: **{m.suitability_category}**",
                f"- **Rationale**: {m.suitability_rationale}",
                "",
            ])

        return "\n".join(lines)

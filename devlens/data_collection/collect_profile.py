import os
import json
import base64
import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from devlens.data_collection.github_client import GitHubClient

logger = logging.getLogger(__name__)


GRAPHQL_PROFILE_QUERY = """
query($username: String!) {
  user(login: $username) {
    login
    name
    bio
    company
    location
    createdAt
    followers {
      totalCount
    }
    following {
      totalCount
    }
    contributionsCollection {
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays {
            date
            contributionCount
            color
          }
        }
      }
    }
    pinnedItems(first: 6, types: REPOSITORY) {
      nodes {
        ... on Repository {
          name
          description
          stargazerCount
          forkCount
          primaryLanguage {
            name
          }
        }
      }
    }
    repositories(first: 100, ownerAffiliations: OWNER, orderBy: {field: STARGAZERS, direction: DESC}) {
      nodes {
        name
        languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
          edges {
            size
            node {
              name
            }
          }
        }
      }
    }
  }
}
"""


class GitHubProfileCollector:
    """Collector for assembling a comprehensive GitHub developer profile."""

    def __init__(self, client: Optional[GitHubClient] = None):
        self.client = client or GitHubClient()

    def collect_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """Collect basic user profile from REST API."""
        return self.client.rest_request(f"users/{username}")

    def collect_public_repos(self, username: str) -> List[Dict[str, Any]]:
        """Collect public repositories for the user via REST API."""
        repos = []
        page = 1
        while True:
            page_repos = self.client.rest_request(
                f"users/{username}/repos",
                params={"type": "public", "per_page": 100, "page": page}
            )
            if not page_repos or not isinstance(page_repos, list):
                break
            
            for repo in page_repos:
                repos.append({
                    "id": repo.get("id"),
                    "name": repo.get("name"),
                    "full_name": repo.get("full_name"),
                    "description": repo.get("description"),
                    "language": repo.get("language"),
                    "stars": repo.get("stargazers_count"),
                    "forks": repo.get("forks_count"),
                    "size": repo.get("size"),
                    "created_at": repo.get("created_at"),
                    "pushed_at": repo.get("pushed_at"),
                    "is_fork": repo.get("fork"),
                    "default_branch": repo.get("default_branch"),
                })
            
            if len(page_repos) < 100:
                break
            page += 1

        return repos

    def _enrich_single_repo(self, username: str, repo: Dict[str, Any], skip_commit_messages: bool = False) -> None:
        """Enrich a single original repo with README, CI, test presence, and recent commit messages."""
        if repo.get("is_fork"):
            repo["readme_length_chars"] = 0
            repo["has_ci_config"] = False
            repo["has_test_presence"] = False
            repo["recent_commit_messages"] = []
            return

        repo_name = repo["name"]

        # 1. README presence and length
        readme_res = self.client.rest_request(f"repos/{username}/{repo_name}/readme", max_retries=2)
        readme_len = 0
        if isinstance(readme_res, dict) and "content" in readme_res:
            try:
                content_bytes = base64.b64decode(readme_res["content"].encode("ascii"))
                readme_len = len(content_bytes.decode("utf-8", errors="ignore"))
            except Exception as e:
                logger.debug(f"Error decoding README for {repo_name}: {e}")
                readme_len = readme_res.get("size", 0)
        repo["readme_length_chars"] = readme_len

        # 2. CI/CD config presence (.github/workflows)
        wf_res = self.client.rest_request(f"repos/{username}/{repo_name}/contents/.github/workflows", max_retries=2)
        has_ci = isinstance(wf_res, list) and len(wf_res) > 0
        if not has_ci:
            for ci_file in (".travis.yml", "circle.yml", "Jenkinsfile", ".gitlab-ci.yml", "azure-pipelines.yml"):
                if self.client.rest_request(f"repos/{username}/{repo_name}/contents/{ci_file}", max_retries=1):
                    has_ci = True
                    break
        repo["has_ci_config"] = has_ci

        # 3. Test directory/file presence (GET /repos/{owner}/{repo}/contents)
        contents_res = self.client.rest_request(f"repos/{username}/{repo_name}/contents", max_retries=2)
        has_tests = False
        if isinstance(contents_res, list):
            test_dir_names = {"tests", "test", "__tests__", "spec", "testing"}
            test_file_patterns = ("test_", "_test.", ".test.", ".spec.")
            for item in contents_res:
                if not isinstance(item, dict):
                    continue
                item_name = item.get("name", "").lower()
                item_type = item.get("type", "")
                if item_type == "dir" and item_name in test_dir_names:
                    has_tests = True
                    break
                elif item_type == "file" and any(p in item_name for p in test_file_patterns):
                    has_tests = True
                    break
        repo["has_test_presence"] = has_tests

        # 4. Recent commit messages (up to 30)
        commit_msgs = []
        if not skip_commit_messages:
            commits_res = self.client.rest_request(
                f"repos/{username}/{repo_name}/commits",
                params={"per_page": 30},
                max_retries=2
            )
            if isinstance(commits_res, list):
                for c_item in commits_res:
                    if isinstance(c_item, dict):
                        raw_msg = c_item.get("commit", {}).get("message", "")
                        subject = raw_msg.split("\n")[0].strip() if raw_msg else ""
                        if subject:
                            commit_msgs.append(subject)
        repo["recent_commit_messages"] = commit_msgs

    def enrich_repo_details(
        self, username: str, repos: List[Dict[str, Any]], skip_commit_messages: bool = False, max_workers: int = 6
    ) -> None:
        """Enrich original (non-fork) repos concurrently using a bounded thread pool."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        non_fork_repos = [r for r in repos if not r.get("is_fork")]
        logger.info(f"Enriching {len(non_fork_repos)} original repositories concurrently (max_workers={max_workers})...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_repo = {
                executor.submit(self._enrich_single_repo, username, repo, skip_commit_messages): repo
                for repo in non_fork_repos
            }
            for future in as_completed(future_to_repo):
                try:
                    future.result()
                except Exception as e:
                    repo_obj = future_to_repo[future]
                    logger.warning(f"Error enriching repo {repo_obj.get('name')}: {e}")

    def collect_commit_activity(self, username: str, repos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect 52-week commit activity for non-fork public repositories."""
        commit_activity_by_repo = {}
        non_fork_repos = [r for r in repos if not r.get("is_fork")]
        
        # Limit to top 15 most recently pushed repos to stay reasonable with API calls
        sorted_repos = sorted(non_fork_repos, key=lambda x: x.get("pushed_at") or "", reverse=True)[:15]
        
        for repo in sorted_repos:
            repo_name = repo["name"]
            stats = self.client.rest_request(f"repos/{username}/{repo_name}/stats/commit_activity", max_retries=2)
            if stats and isinstance(stats, list):
                commit_activity_by_repo[repo_name] = stats

        return commit_activity_by_repo

    def collect_pr_history(self, username: str) -> Dict[str, Any]:
        """Collect PR metrics (opened count, merged count, review participation)."""
        pr_opened = self.client.rest_request(
            "search/issues",
            params={"q": f"author:{username} type:pr", "per_page": 1}
        )
        pr_merged = self.client.rest_request(
            "search/issues",
            params={"q": f"author:{username} type:pr is:merged", "per_page": 1}
        )
        pr_reviewed = self.client.rest_request(
            "search/issues",
            params={"q": f"reviewed-by:{username} type:pr", "per_page": 1}
        )

        return {
            "total_opened": pr_opened.get("total_count", 0) if pr_opened else 0,
            "total_merged": pr_merged.get("total_count", 0) if pr_merged else 0,
            "review_participation_count": pr_reviewed.get("total_count", 0) if pr_reviewed else 0,
        }

    def collect_issue_activity(self, username: str) -> Dict[str, Any]:
        """Collect issue metrics (opened count, closed count, comment activity)."""
        issues_opened = self.client.rest_request(
            "search/issues",
            params={"q": f"author:{username} type:issue", "per_page": 1}
        )
        issues_closed = self.client.rest_request(
            "search/issues",
            params={"q": f"author:{username} type:issue is:closed", "per_page": 1}
        )
        issues_commented = self.client.rest_request(
            "search/issues",
            params={"q": f"commenter:{username} type:issue", "per_page": 1}
        )

        return {
            "total_opened": issues_opened.get("total_count", 0) if issues_opened else 0,
            "total_closed": issues_closed.get("total_count", 0) if issues_closed else 0,
            "total_commented": issues_commented.get("total_count", 0) if issues_commented else 0,
        }

    def collect_graphql_data(self, username: str) -> Optional[Dict[str, Any]]:
        """Collect nested profile data using GraphQL API v4."""
        gql_data = self.client.graphql_request(GRAPHQL_PROFILE_QUERY, variables={"username": username})
        if not gql_data or "user" not in gql_data or not gql_data["user"]:
            return None

        user_gql = gql_data["user"]
        
        # Contribution calendar
        cal_data = user_gql.get("contributionsCollection", {}).get("contributionCalendar", {})
        
        # Pinned repos
        pinned_nodes = user_gql.get("pinnedItems", {}).get("nodes", [])
        pinned_repos = [
            {
                "name": node.get("name"),
                "description": node.get("description"),
                "stars": node.get("stargazerCount"),
                "forks": node.get("forkCount"),
                "primary_language": node.get("primaryLanguage", {}).get("name") if node.get("primaryLanguage") else None,
            }
            for node in pinned_nodes if node
        ]

        # Language breakdown per repo
        repo_nodes = user_gql.get("repositories", {}).get("nodes", [])
        languages_breakdown = {}
        for r_node in repo_nodes:
            if not r_node:
                continue
            r_name = r_node.get("name")
            lang_edges = r_node.get("languages", {}).get("edges", [])
            languages_breakdown[r_name] = [
                {"language": edge["node"]["name"], "size_bytes": edge["size"]}
                for edge in lang_edges if edge and "node" in edge
            ]

        return {
            "contribution_calendar": cal_data,
            "pinned_repositories": pinned_repos,
            "languages_breakdown": languages_breakdown,
        }

    def collect_full_profile(self, username: str, skip_commit_messages: bool = False) -> Dict[str, Any]:
        """Orchestrate collection of REST and GraphQL data into a unified profile dictionary."""
        logger.info(f"Starting data collection for GitHub user: '{username}'")

        user_info = self.collect_user_info(username)
        if not user_info:
            logger.error(f"User '{username}' not found or REST API request failed.")
            raise ValueError(f"Could not retrieve user info for '{username}'.")

        repos = self.collect_public_repos(username)
        self.enrich_repo_details(username, repos, skip_commit_messages=skip_commit_messages)
        commit_activity = self.collect_commit_activity(username, repos)
        pr_history = self.collect_pr_history(username)
        issue_activity = self.collect_issue_activity(username)
        graphql_data = self.collect_graphql_data(username)

        # Calculate total commits from GraphQL calendar if available, else sum commit activity
        total_commits = 0
        if graphql_data and "contribution_calendar" in graphql_data:
            total_commits = graphql_data["contribution_calendar"].get("totalContributions", 0)
        else:
            for stats in commit_activity.values():
                if isinstance(stats, list):
                    total_commits += sum(w.get("total", 0) for w in stats)

        summary_metrics = {
            "repo_count": len(repos),
            "total_commits": total_commits,
            "api_calls_used": self.client.get_api_calls_summary(),
        }

        profile_data = {
            "username": username,
            "collected_at": datetime.utcnow().isoformat() + "Z",
            "user_info": user_info,
            "repositories": repos,
            "commit_activity": commit_activity,
            "pull_requests": pr_history,
            "issues": issue_activity,
            "graphql_data": graphql_data,
            "summary_metrics": summary_metrics,
        }

        return profile_data

    def save_profile_data(self, profile_data: Dict[str, Any], output_dir: str = "data/raw") -> str:
        """Save collected profile dictionary to data/raw/{username}.json."""
        username = profile_data["username"]
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{username}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved raw profile data to: {file_path}")
        return file_path

    def collect_batch_profiles(
        self,
        usernames_file: str,
        output_dir: str = "data/raw",
        skip_commit_messages: bool = False,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Collect profile data sequentially for a batch of usernames."""
        import time
        import numpy as np

        start_time = time.time()
        today_str = datetime.utcnow().strftime("%Y-%m-%d")

        if not os.path.exists(usernames_file):
            raise FileNotFoundError(f"Batch input file not found: '{usernames_file}'")

        with open(usernames_file, "r", encoding="utf-8") as f:
            usernames = [line.strip() for line in f if line.strip()]

        total_input_usernames = len(usernames)
        if limit and limit > 0:
            usernames = usernames[:limit]

        total_to_process = len(usernames)
        logger.info(f"Starting batch profile collection for {total_to_process} users (limit={limit}, skip_commit_messages={skip_commit_messages})...")

        succeeded_count = 0
        failed_count = 0
        skipped_count = 0
        failed_users = {}
        processed_repo_counts = []
        user_api_call_counts = []

        initial_api_calls = self.client.api_calls_count

        for idx, uname in enumerate(usernames, start=1):
            file_path = os.path.join(output_dir, f"{uname}.json")

            # Resumable check: skip if collected today
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as rf:
                        existing_data = json.load(rf)
                    collected_at = existing_data.get("collected_at", "")
                    if collected_at.startswith(today_str):
                        skipped_count += 1
                        logger.info(f"[{idx}/{total_to_process}] {uname} - SKIPPED (already collected today)")
                        continue
                except Exception:
                    pass  # If existing JSON is corrupt, re-fetch

            user_start_calls = self.client.api_calls_count
            try:
                profile = self.collect_full_profile(uname, skip_commit_messages=skip_commit_messages)
                self.save_profile_data(profile, output_dir=output_dir)

                user_calls = self.client.api_calls_count - user_start_calls
                repo_count = profile["summary_metrics"]["repo_count"]
                remaining = self.client.last_rate_limit_remaining

                processed_repo_counts.append(repo_count)
                user_api_call_counts.append(user_calls)
                succeeded_count += 1

                rem_str = str(remaining) if remaining is not None else "N/A"
                print(f"[{idx}/{total_to_process}] {uname} - repos: {repo_count}, API calls used: {user_calls}, remaining: {rem_str}")

            except Exception as e:
                failed_count += 1
                failed_users[uname] = str(e)
                logger.error(f"[{idx}/{total_to_process}] {uname} - FAILED: {e}")

        elapsed = round(time.time() - start_time, 2)
        total_calls_consumed = self.client.api_calls_count - initial_api_calls

        batch_summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input_file": usernames_file,
            "total_input_usernames": total_input_usernames,
            "processed_count": total_to_process,
            "skipped_count": skipped_count,
            "succeeded_count": succeeded_count,
            "failed_count": failed_count,
            "failed_users": failed_users,
            "total_api_calls_consumed": total_calls_consumed,
            "elapsed_seconds": elapsed,
            "avg_repos_per_user": round(float(np.mean(processed_repo_counts)), 2) if processed_repo_counts else 0.0,
            "avg_api_calls_per_user": round(float(np.mean(user_api_call_counts)), 2) if user_api_call_counts else 0.0,
        }

        summary_path = os.path.join(output_dir, "batch_summary.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Batch execution finished. Summary saved to '{summary_path}'.")
        return batch_summary


def main():
    parser = argparse.ArgumentParser(description="Collect GitHub profile data for a developer or batch of developers.")
    parser.add_argument("username", type=str, nargs="?", default=None, help="GitHub username to profile")
    parser.add_argument("--batch", type=str, default=None, help="Path to text file containing usernames (one per line)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of users to process in batch mode")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory for raw JSON")
    parser.add_argument(
        "--skip-commit-messages",
        action="store_true",
        help="Skip fetching recent commit messages to reduce API calls during bulk collection",
    )
    args = parser.parse_args()

    client = GitHubClient()
    collector = GitHubProfileCollector(client=client)

    if args.batch:
        summary = collector.collect_batch_profiles(
            usernames_file=args.batch,
            output_dir=args.output_dir,
            skip_commit_messages=args.skip_commit_messages,
            limit=args.limit
        )
        print("\n" + "=" * 60)
        print("  DEV LENS - BATCH COLLECTION SUMMARY")
        print("=" * 60)
        print(f" Input File             : {summary['input_file']}")
        print(f" Total Input Usernames  : {summary['total_input_usernames']}")
        print(f" Processed Count        : {summary['processed_count']}")
        print(f" Skipped (Today)        : {summary['skipped_count']}")
        print(f" Succeeded Count        : {summary['succeeded_count']}")
        print(f" Failed Count           : {summary['failed_count']}")
        print(f" Total API Calls        : {summary['total_api_calls_consumed']}")
        print(f" Avg Repos per User     : {summary['avg_repos_per_user']}")
        print(f" Avg API Calls / User   : {summary['avg_api_calls_per_user']}")
        print(f" Elapsed Time           : {summary['elapsed_seconds']} seconds")
        print(f" Summary Output Path    : {os.path.join(args.output_dir, 'batch_summary.json')}")
        print("=" * 60 + "\n")

    elif args.username:
        try:
            profile = collector.collect_full_profile(args.username, skip_commit_messages=args.skip_commit_messages)
            saved_path = collector.save_profile_data(profile, output_dir=args.output_dir)

            api_summary = profile["summary_metrics"]["api_calls_used"]
            print("\n" + "=" * 50)
            print(f"  DEV LENS - PROFILE COLLECTION SUMMARY ({args.username})")
            print("=" * 50)
            print(f" Username        : {profile['username']}")
            print(f" Repositories    : {profile['summary_metrics']['repo_count']}")
            print(f" Total Commits   : {profile['summary_metrics']['total_commits']}")
            print(f" API Calls Used  : {api_summary['total_calls']} (REST: {api_summary['rest_calls']}, GraphQL: {api_summary['graphql_calls']})")
            print(f" Saved Output    : {saved_path}")
            print("=" * 50 + "\n")

        except Exception as e:
            logger.error(f"Collection failed: {e}")
            raise
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

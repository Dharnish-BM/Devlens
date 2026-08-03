import os
import json
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

    def collect_full_profile(self, username: str) -> Dict[str, Any]:
        """Orchestrate collection of REST and GraphQL data into a unified profile dictionary."""
        logger.info(f"Starting data collection for GitHub user: '{username}'")

        user_info = self.collect_user_info(username)
        if not user_info:
            logger.error(f"User '{username}' not found or REST API request failed.")
            raise ValueError(f"Could not retrieve user info for '{username}'.")

        repos = self.collect_public_repos(username)
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


def main():
    parser = argparse.ArgumentParser(description="Collect GitHub profile data for a developer.")
    parser.add_argument("username", type=str, help="GitHub username to profile")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory for raw JSON")
    args = parser.parse_args()

    client = GitHubClient()
    collector = GitHubProfileCollector(client=client)

    try:
        profile = collector.collect_full_profile(args.username)
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


if __name__ == "__main__":
    main()

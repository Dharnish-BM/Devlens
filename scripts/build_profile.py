import requests
import os
from dotenv import load_dotenv
import pandas as pd
import time

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")

REST_HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json"
}
GRAPHQL_HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}
GRAPHQL_URL = "https://api.github.com/graphql"

CONTRIB_QUERY = """
query($username: String!) {
  user(login: $username) {
    contributionsCollection {
      totalCommitContributions
      totalPullRequestContributions
      totalIssueContributions
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays {
            date
            contributionCount
          }
        }
      }
    }
  }
}
"""

def get_basic_profile(username):
    url = f"https://api.github.com/users/{username}"
    r = requests.get(url, headers=REST_HEADERS)
    if r.status_code != 200:
        return None
    return r.json()

def get_repos(username):
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos"
        params = {"per_page": 100, "page": page, "type": "owner"}
        r = requests.get(url, headers=REST_HEADERS, params=params)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        repos.extend(data)
        page += 1
    return repos

def get_contributions(username):
    payload = {"query": CONTRIB_QUERY, "variables": {"username": username}}
    r = requests.post(GRAPHQL_URL, headers=GRAPHQL_HEADERS, json=payload)
    if r.status_code != 200:
        return None
    return r.json()

def build_single_profile(username):
    print(f"Fetching: {username}...")
    
    profile = get_basic_profile(username)
    if not profile:
        print(f"  Skipped {username} (profile not found or API error)")
        return None
    
    repos = get_repos(username)
    contrib_data = get_contributions(username)
    
    # --- Repo-level aggregation ---
    total_stars = sum(r["stargazers_count"] for r in repos)
    total_forks = sum(r["forks_count"] for r in repos)
    non_fork_repos = [r for r in repos if not r["fork"]]
    all_languages = set()
    for r in repos:
        if r.get("language"):
            all_languages.add(r["language"])
    
    # --- Contribution / consistency aggregation ---
    commit_contribs = 0
    pr_contribs = 0
    issue_contribs = 0
    consistency_percent = 0
    
    if contrib_data and contrib_data.get("data") and contrib_data["data"].get("user"):
        contrib = contrib_data["data"]["user"]["contributionsCollection"]
        commit_contribs = contrib["totalCommitContributions"]
        pr_contribs = contrib["totalPullRequestContributions"]
        issue_contribs = contrib["totalIssueContributions"]
        
        active_days, total_days = 0, 0
        for week in contrib["contributionCalendar"]["weeks"]:
            for day in week["contributionDays"]:
                total_days += 1
                if day["contributionCount"] > 0:
                    active_days += 1
        consistency_percent = round((active_days / total_days) * 100, 2) if total_days else 0
    
    row = {
        "username": username,
        "name": profile.get("name"),
        "bio": profile.get("bio"),
        "public_repos": profile.get("public_repos"),
        "followers": profile.get("followers"),
        "following": profile.get("following"),
        "account_created": profile.get("created_at"),
        "total_repos_fetched": len(repos),
        "non_fork_repos": len(non_fork_repos),
        "total_stars": total_stars,
        "total_forks": total_forks,
        "num_unique_languages": len(all_languages),
        "languages": ", ".join(all_languages),
        "commit_contributions_past_year": commit_contribs,
        "pr_contributions_past_year": pr_contribs,
        "issue_contributions_past_year": issue_contribs,
        "consistency_score": consistency_percent
    }
    
    return row

if __name__ == "__main__":
    test_usernames = ["torvalds", "gaearon", "sindresorhus"]  # small test batch
    
    all_rows = []
    for username in test_usernames:
        row = build_single_profile(username)
        if row:
            all_rows.append(row)
        time.sleep(1)  # be polite between full profile pulls
    
    df = pd.DataFrame(all_rows)
    df.to_csv("test_batch_profiles.csv", index=False)
    print(f"\nSaved {len(df)} profiles to test_batch_profiles.csv")
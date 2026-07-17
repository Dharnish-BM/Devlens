import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")

GRAPHQL_URL = "https://api.github.com/graphql"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# This query asks GitHub for one full year of daily contribution counts,
# plus total commits, PRs, and issues — all in a single request.
QUERY = """
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

def get_contribution_data(username):
    payload = {
        "query": QUERY,
        "variables": {"username": username}
    }
    response = requests.post(GRAPHQL_URL, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def summarize_consistency(data, username):
    """Turn the raw daily data into useful consistency numbers."""
    contrib = data["data"]["user"]["contributionsCollection"]
    calendar = contrib["contributionCalendar"]["weeks"]
    
    total_days_active = 0
    total_days_in_year = 0
    
    for week in calendar:
        for day in week["contributionDays"]:
            total_days_in_year += 1
            if day["contributionCount"] > 0:
                total_days_active += 1
    
    consistency_percent = round((total_days_active / total_days_in_year) * 100, 2)
    
    print(f"\n--- {username} ---")
    print("Total Commit Contributions (past year):", contrib["totalCommitContributions"])
    print("Total PR Contributions:", contrib["totalPullRequestContributions"])
    print("Total Issue Contributions:", contrib["totalIssueContributions"])
    print("Days Active:", total_days_active, "/", total_days_in_year)
    print("Consistency Score:", consistency_percent, "%")

if __name__ == "__main__":
    username = "torvalds"
    data = get_contribution_data(username)
    
    if data:
        summarize_consistency(data, username)
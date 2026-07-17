import requests
import os
from dotenv import load_dotenv
import pandas as pd
import time

load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json"
}

def get_repos(username):
    """Fetch all public repos for a user (handles pagination)."""
    repos = []
    page = 1
    while True:
        url = f"https://api.github.com/users/{username}/repos"
        params = {"per_page": 100, "page": page, "type": "owner", "sort": "updated"}
        response = requests.get(url, headers=HEADERS, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching repos: {response.status_code}")
            break
        
        data = response.json()
        if not data:  # empty page = no more repos
            break
        
        repos.extend(data)
        page += 1
    
    return repos

def get_languages(username, repo_name):
    """Fetch languages used in a specific repo."""
    url = f"https://api.github.com/repos/{username}/{repo_name}/languages"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()  # e.g. {"Python": 5000, "HTML": 1200}
    return {}

def build_repo_dataset(username):
    repos = get_repos(username)
    rows = []
    
    for repo in repos:
        repo_name = repo["name"]
        languages = get_languages(username, repo_name)
        
        rows.append({
            "username": username,
            "repo_name": repo_name,
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "size_kb": repo["size"],
            "created_at": repo["created_at"],
            "updated_at": repo["updated_at"],
            "is_fork": repo["fork"],
            "languages": ", ".join(languages.keys()) if languages else "None",
            "num_languages": len(languages)
        })
        
        time.sleep(0.2)  # small delay to be polite to the API
    
    return pd.DataFrame(rows)

if __name__ == "__main__":
    username = "torvalds"
    df = build_repo_dataset(username)
    print(df)
    
    # Save to CSV
    df.to_csv(f"{username}_repos.csv", index=False)
    print(f"\nSaved {len(df)} repos to {username}_repos.csv")
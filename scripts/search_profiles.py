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

SEARCH_URL = "https://api.github.com/search/users"

def search_users(query, max_pages=10):
    """Search GitHub users matching a query, up to max_pages (100 results/page)."""
    all_users = []
    
    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "per_page": 100,
            "page": page
        }
        response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
        
        if response.status_code != 200:
            print(f"  Error {response.status_code}: {response.text[:200]}")
            break
        
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            break  # no more results
        
        all_users.extend(items)
        print(f"  Page {page}: got {len(items)} users (total so far: {len(all_users)})")
        
        # Search API rate limit is strict: 30 requests/minute
        time.sleep(2.5)
    
    return all_users

def build_username_list():
    """Run multiple search queries to build a diverse pool of usernames."""
    
    # Different queries = different slices of the population
    # Splitting by created-date ranges helps us get past the 1000-result cap per query
    queries = [
        "location:India followers:<100 repos:2..50 created:2021-01-01..2022-06-30",
        "location:India followers:<100 repos:2..50 created:2022-07-01..2023-12-31",
        "location:India followers:<100 repos:2..50 created:2024-01-01..2024-12-31",
    ]
    
    all_usernames = set()  # set avoids duplicates automatically
    
    for q in queries:
        print(f"\nRunning query: {q}")
        users = search_users(q, max_pages=10)
        for u in users:
            all_usernames.add(u["login"])
    
    return list(all_usernames)

if __name__ == "__main__":
    usernames = build_username_list()
    print(f"\n✅ Total unique usernames collected: {len(usernames)}")
    
    df = pd.DataFrame({"username": usernames})
    df.to_csv("supplementary_usernames.csv", index=False)
    print("Saved to supplementary_usernames.csv")
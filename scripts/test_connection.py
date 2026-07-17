import requests
import os
from dotenv import load_dotenv

# Load the token from .env
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")

# Headers required by GitHub API for authentication
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json"
}

def get_user_profile(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        print("Username:", data.get("login"))
        print("Name:", data.get("name"))
        print("Public Repos:", data.get("public_repos"))
        print("Followers:", data.get("followers"))
        print("Account Created:", data.get("created_at"))
    else:
        print(f"Error {response.status_code}: {response.text}")

def check_rate_limit():
    url = "https://api.github.com/rate_limit"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    remaining = data["rate"]["remaining"]
    limit = data["rate"]["limit"]
    print(f"\nAPI calls remaining: {remaining}/{limit}")

if __name__ == "__main__":
    get_user_profile("torvalds")  # testing with Linus Torvalds' profile
    check_rate_limit()
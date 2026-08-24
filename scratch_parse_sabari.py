import json

filepath = 'data/raw/Sabari-Vasan-SM.json'
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# The structure of the JSON is determined by Phase 1 data collection.
created_at = data.get('created_at', 'Unknown')
repos = data.get('repositories', [])
total_repo_count = len(repos)

# Extract top 3 repos by size
sorted_repos = sorted(repos, key=lambda x: x.get('size', 0) if isinstance(x, dict) else 0, reverse=True)
top_3_repos = sorted_repos[:3]

# Extract PRs
prs = data.get('pull_requests', [])
if isinstance(prs, dict):
    pr_opened = len(prs.keys())
    pr_merged = sum(1 for k, v in prs.items() if isinstance(v, dict) and (v.get('merged_at') or v.get('state') == 'merged' or v.get('is_merged')))
elif isinstance(prs, list):
    pr_opened = len(prs)
    pr_merged = 0
    for pr in prs:
        if isinstance(pr, dict):
            if pr.get('merged_at') or pr.get('state') == 'merged' or pr.get('is_merged'):
                pr_merged += 1
else:
    pr_opened = 'Unknown format'
    pr_merged = 'Unknown format'
    print("PR format:", type(prs))

print(f'Account Creation Date: {created_at}')
print(f'Total Repo Count: {total_repo_count}')
print('Top 3 Repos by Size:')
for r in top_3_repos:
    if isinstance(r, dict):
        print(f"  - {r.get('name')}: {r.get('size')} KB")
    else:
        print(f"  - Unknown format: {r}")

print(f'Total PRs Opened: {pr_opened}')
print(f'Total PRs Merged: {pr_merged}')

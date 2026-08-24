import sys
import json
sys.path.append('d:/GIT/Devlens')
from devlens.db.session import get_session
from devlens.db.repository import get_all_current_features
import pandas as pd

# 1. Raw JSON inspection for AravindParamasivan & riyazhini2005
print("=== 1. RAW JSON INSPECTION ===")
for user in ["AravindParamasivan", "riyazhini2005"]:
    path = f"data/raw/{user}.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        repos = data.get("repositories", [])
        gql = data.get("graphql_data", {}).get("languages_breakdown", {})
        print(f"\n--- User: {user} ---")
        print(f"Total Repos: {len(repos)}")
        for r in repos:
            print(f"  Repo: {r.get('name')} | Primary Lang: {r.get('language')} | Fork: {r.get('is_fork')} | Size: {r.get('size')} KB")
        print(f"GraphQL Language Breakdown: {gql}")
    except Exception as e:
        print(f"Error reading {path}: {e}")

# 2 & 3. Full 7-Label Distribution Analysis
print("\n" + "="*80)
print("=== 2 & 3. HIERARCHICAL 7-LABEL CLASSIFICATION DISTRIBUTION ===")
print("="*80)

with get_session() as session:
    df = get_all_current_features(session)

labels = []
for username, row in df.iterrows():
    ml_sig = row.get("lang_ml_signal", 0.0)
    mobile_sig = row.get("lang_mobile_signal", 0.0)
    devops_sig = row.get("lang_devops_signal", 0.0)
    frontend_sig = row.get("lang_frontend_signal", 0.0)
    primary_js = row.get("lang_primary_is_javascript", 0.0) == 1.0
    primary_c = row.get("lang_primary_is_c", 0.0) == 1.0
    primary_java_kotlin = row.get("lang_primary_is_java_kotlin", 0.0) == 1.0
    entropy = row.get("lang_diversity_entropy", 0.0)

    # Hierarchical Evaluation
    if ml_sig > 0.05:
        lbl = "ML Specialist"
    elif mobile_sig > 0.02:
        lbl = "Mobile Developer"
    elif devops_sig > 0.10:
        lbl = "DevOps Engineer"
    elif frontend_sig > 0.50 or primary_js:
        lbl = "Frontend Developer"
    elif primary_c or (primary_java_kotlin and mobile_sig <= 0.02):
        lbl = "Backend Developer"
    elif entropy >= 1.8:
        lbl = "Full-Stack Developer"
    else:
        lbl = "Unclassified / Low Signal"
    
    labels.append(lbl)

df["archetype_label"] = labels

dist = df["archetype_label"].value_counts()
print(f"\nTotal Cohort: {len(df)}")
print("-" * 50)
print(f"{'Archetype Label':<30} {'Count':<10} {'Percentage':<10}")
print("-" * 50)
for lbl, count in dist.items():
    pct = count / len(df) * 100
    print(f"{lbl:<30} {count:<10} {pct:.2f}%")
print("-" * 50)

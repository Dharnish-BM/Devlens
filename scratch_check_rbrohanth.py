import sys
import json
sys.path.append('d:/GIT/Devlens')
from devlens.db.session import get_session
from devlens.db.repository import get_all_current_features
from devlens.db.models import Developer, Snapshot, DocScore, EngMaturityScore, ArchetypePrediction

# 1. Inspect DB entries
with get_session() as session:
    dev = session.query(Developer).filter(Developer.username == "RBROHANTH").first()
    snap = dev.snapshots[-1] if dev else None
    
    doc = session.query(DocScore).filter(DocScore.snapshot_id == snap.id).first() if snap else None
    eng = session.query(EngMaturityScore).filter(EngMaturityScore.snapshot_id == snap.id).first() if snap else None
    arch = session.query(ArchetypePrediction).filter(ArchetypePrediction.snapshot_id == snap.id).first() if snap else None
    
    print("=== DB PROFILE FOR RBROHANTH ===")
    print(f"Username:               {dev.username if dev else 'None'}")
    print(f"Archetype Label:        {arch.archetype_label if arch else 'None'}")
    print(f"Confidence:             {arch.confidence if arch else 'None'}")
    print(f"Doc Score:              {doc.score if doc else 'None'}")
    print(f"Doc Components:         {doc.components if doc else 'None'}")
    print(f"Eng Maturity Score:     {eng.score if eng else 'None'}")
    print(f"Eng Maturity Components:{eng.components if eng else 'None'}")

# 2. Inspect Raw JSON
with open("data/raw/RBROHANTH.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

repos = raw.get("repositories", [])
user_info = raw.get("user_info", {})
gql = raw.get("graphql_data", {})
lb = gql.get("languages_breakdown", {})

print("\n=== RAW JSON SUMMARY FOR RBROHANTH ===")
print(f"Name / Bio:             {user_info.get('name')} | {user_info.get('bio')}")
print(f"Public Repos Count:     {user_info.get('public_repos')}")
print(f"Total Repos in JSON:    {len(repos)}")

original_repos = [r for r in repos if not r.get("is_fork")]
fork_repos = [r for r in repos if r.get("is_fork")]
print(f"Original Repos:         {len(original_repos)}")
print(f"Forked Repos:           {len(fork_repos)}")

print("\nRepository Breakdown:")
for r in repos:
    print(f"  • {r.get('name'):<30} | Lang: {str(r.get('language')):<12} | Fork: {str(r.get('is_fork')):<5} | Size: {r.get('size')} KB | CI: {r.get('has_ci_config')}")

lang_bytes = {}
for repo_langs in lb.values():
    if isinstance(repo_langs, list):
        for entry in repo_langs:
            if isinstance(entry, dict):
                lang = entry.get("language", "Unknown")
                lang_bytes[lang] = lang_bytes.get(lang, 0) + (entry.get("size_bytes", 0) or 0)

tot_bytes = sum(lang_bytes.values())
print("\nLanguage Byte Distribution:")
for l, b in sorted(lang_bytes.items(), key=lambda x: -x[1]):
    print(f"  • {l:<20}: {b:>10,d} bytes ({b/tot_bytes*100:>5.1f}%)")

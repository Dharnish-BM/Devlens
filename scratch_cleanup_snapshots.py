import sys
import json
from datetime import datetime
from pathlib import Path
sys.path.append('d:/GIT/Devlens')
from devlens.db.session import get_session
from devlens.db.models import Developer, Snapshot, Feature, ClusterAssignment, ArchetypePrediction, DocScore, EngMaturityScore
from devlens.models.growth_trajectory import get_growth_summary

raw_dir = Path("data/raw")

print("=== 1. RUNNING DATABASE SNAPSHOT CLEANUP ===")
with get_session() as session:
    devs = session.query(Developer).all()
    print(f"Total Developers: {len(devs)}")
    
    total_deleted = 0
    retained_snapshots = []
    
    for dev in devs:
        snaps = session.query(Snapshot).filter_by(developer_id=dev.id).order_by(Snapshot.id.asc()).all()
        if not snaps:
            continue
        
        # Keep the latest snapshot
        latest_snap = snaps[-1]
        retained_snapshots.append(latest_snap.id)
        
        # Update collected_at timestamp from raw JSON if available
        json_path = raw_dir / f"{dev.username}.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                meta_collected = raw_data.get("collected_at")
                if meta_collected:
                    # ISO string to datetime
                    # e.g. "2026-08-13T03:48:26Z"
                    clean_ts = meta_collected.replace("Z", "+00:00")
                    latest_snap.collected_at = datetime.fromisoformat(clean_ts)
                latest_snap.raw_json_path = str(json_path)
            except Exception as e:
                pass
        
        # Delete older snapshots for this developer
        older_snaps = snaps[:-1]
        for old_s in older_snaps:
            session.delete(old_s)
            total_deleted += 1
            
    session.commit()
    print(f"Cleanup complete: Deleted {total_deleted} intermediate snapshot rows.")

# 2. Verify Database State
with get_session() as session:
    total_devs = session.query(Developer).count()
    total_snaps = session.query(Snapshot).count()
    total_features = session.query(Feature).count()
    total_clusters = session.query(ClusterAssignment).count()
    total_archetypes = session.query(ArchetypePrediction).count()
    total_docs = session.query(DocScore).count()
    total_engs = session.query(EngMaturityScore).count()
    
    print("\n=== 2. POST-CLEANUP DATABASE COUNTS ===")
    print(f"Total Developers:             {total_devs}")
    print(f"Total Snapshots:              {total_snaps} (Exactly 1 per developer: {total_snaps == total_devs})")
    print(f"Total Features:               {total_features} ({total_features/total_snaps:.1f} per snapshot)")
    print(f"Total Cluster Assignments:   {total_clusters}")
    print(f"Total Archetype Predictions:  {total_archetypes}")
    print(f"Total Doc Scores:             {total_docs}")
    print(f"Total Eng Maturity Scores:    {total_engs}")

# 3. Test get_growth_summary() on 5 real users
print("\n" + "="*80)
print("=== 3. GROWTH TRAJECTORY TEST ON 5 REAL USERS (POST-CLEANUP) ===")
print("="*80)

test_users = [
    "dineshkumarp07",
    "vijeth06",
    "Jeevashre12",
    "AMARNATH002",
    "RBROHANTH",
]

for uname in test_users:
    res = get_growth_summary(uname)
    print(f"\nDeveloper: {uname}")
    print(f"  • Status:         {res.get('status')}")
    print(f"  • Snapshot Count: {res.get('snapshot_count')}")
    print(f"  • Message:        {res.get('message')}")

print("\n" + "="*80 + "\n")

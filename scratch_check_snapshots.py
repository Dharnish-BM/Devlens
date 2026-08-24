import sys
sys.path.append('d:/GIT/Devlens')
from devlens.db.session import get_session
from devlens.db.models import Developer, Snapshot, Feature
from devlens.models.growth_trajectory import get_growth_summary

with get_session() as session:
    dev = session.query(Developer).filter_by(username="dineshkumarp07").first()
    snaps = session.query(Snapshot).filter_by(developer_id=dev.id).order_by(Snapshot.collected_at.asc()).all()
    print(f"Snapshots for {dev.username}: {len(snaps)}")
    for s in snaps:
        fc = session.query(Feature).filter_by(snapshot_id=s.id).count()
        print(f"  • Snapshot ID: {s.id:<3} | Collected At: {s.collected_at} | Feature count: {fc}")

print("\nGrowth summary for dineshkumarp07:")
summary = get_growth_summary("dineshkumarp07")
print(f"Status: {summary['status']}")
print(f"Timespan days: {summary.get('timespan_days'):.4f}")
print(f"Archetype stability: {summary.get('archetype_stability')}")
print(f"Doc score slope/month: {summary.get('doc_score_slope_per_month')}")

# Test with a single snapshot user (or create a temporary single-snapshot user)
temp_user = "test_single_snapshot_student"
with get_session() as session:
    # Check if exists
    existing = session.query(Developer).filter_by(username=temp_user).first()
    if not existing:
        t_dev = Developer(username=temp_user)
        session.add(t_dev)
        session.flush()
        t_snap = Snapshot(developer_id=t_dev.id)
        session.add(t_snap)
        session.commit()

single_summary = get_growth_summary(temp_user)
print(f"\nSingle-Snapshot Student Test ({temp_user}):")
print(f"Status: {single_summary['status']}")
print(f"Snapshot Count: {single_summary['snapshot_count']}")
print(f"Message: {single_summary['message']}")

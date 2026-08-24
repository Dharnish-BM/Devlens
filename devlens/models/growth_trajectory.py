"""
DevLens Phase 7: Growth Trajectory & Longitudinal Trend Analysis.

- Analyzes developer progression across longitudinal collection snapshots
- Evaluates feature-level deltas and linear growth rates over time
- Tracks archetype stability across snapshot history
- Returns explicit 'insufficient_history' status when only 1 snapshot exists without fabricating trends
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from devlens.db.models import ArchetypePrediction, Developer, DocScore, EngMaturityScore, Feature, Snapshot
from devlens.db.repository import get_developer, get_snapshot_history
from devlens.db.session import get_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_linear_slope(times_days: List[float], values: List[float]) -> float:
    """Compute simple linear regression slope (units change per day)."""
    if len(times_days) < 2 or len(set(times_days)) <= 1:
        return 0.0
    x = np.array(times_days)
    y = np.array(values)
    # y = slope * x + intercept
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def get_growth_summary(username: str, session: Optional[Session] = None) -> Dict[str, Any]:
    """Retrieve growth trajectory and longitudinal trend analysis for a developer.
    
    Args:
        username: GitHub username
        session: Optional SQLAlchemy Session
        
    Returns:
        Dict with growth summary metrics or 'insufficient_history' if < 2 snapshots.
    """
    if session is None:
        with get_session() as s:
            return get_growth_summary(username, session=s)

    history = get_snapshot_history(session, username)

    if not history:
        return {
            "status": "not_found",
            "message": f"Developer '{username}' not found or has no snapshots.",
            "snapshot_count": 0,
        }

    if len(history) == 1:
        return {
            "status": "insufficient_history",
            "message": "Only one collection snapshot available; trend analysis requires repeat collections over time.",
            "snapshot_count": 1,
        }

    # If 2+ snapshots exist, compute longitudinal metrics
    dev = get_developer(session, username)
    snapshots = (
        session.query(Snapshot)
        .filter_by(developer_id=dev.id)
        .order_by(Snapshot.collected_at.asc())
        .all()
    )

    t0 = snapshots[0].collected_at
    times_days = [(s.collected_at - t0).total_seconds() / 86400.0 for s in snapshots]

    doc_scores = []
    eng_scores = []
    archetypes = []
    snapshot_details = []

    for s in snapshots:
        doc_obj = session.query(DocScore).filter_by(snapshot_id=s.id).first()
        eng_obj = session.query(EngMaturityScore).filter_by(snapshot_id=s.id).first()
        arch_obj = session.query(ArchetypePrediction).filter_by(snapshot_id=s.id).first()

        d_val = doc_obj.score if doc_obj else 0.0
        e_val = eng_obj.score if eng_obj else 0.0
        a_val = arch_obj.archetype_label if arch_obj else "Unclassified / Low Signal"

        doc_scores.append(d_val)
        eng_scores.append(e_val)
        archetypes.append(a_val)

        snapshot_details.append({
            "snapshot_id": s.id,
            "collected_at": s.collected_at.isoformat(),
            "days_from_start": (s.collected_at - t0).total_seconds() / 86400.0,
            "doc_score": d_val,
            "eng_maturity_score": e_val,
            "archetype_label": a_val,
        })

    # Compute slopes (change per 30-day month)
    doc_slope_day = compute_linear_slope(times_days, doc_scores)
    eng_slope_day = compute_linear_slope(times_days, eng_scores)
    doc_slope_month = doc_slope_day * 30.0
    eng_slope_month = eng_slope_day * 30.0

    # Feature-level deltas between earliest and latest snapshot
    earliest_id = snapshots[0].id
    latest_id = snapshots[-1].id

    earliest_feats = {
        f.feature_name: f.feature_value
        for f in session.query(Feature).filter_by(snapshot_id=earliest_id).all()
    }
    latest_feats = {
        f.feature_name: f.feature_value
        for f in session.query(Feature).filter_by(snapshot_id=latest_id).all()
    }

    feature_deltas = {}
    for feat_name, lat_val in latest_feats.items():
        if feat_name in earliest_feats and lat_val is not None and earliest_feats[feat_name] is not None:
            feature_deltas[feat_name] = float(lat_val - earliest_feats[feat_name])

    # Archetype stability
    is_stable = len(set(archetypes)) == 1

    return {
        "status": "sufficient_history",
        "username": username,
        "snapshot_count": len(snapshots),
        "first_collected_at": snapshots[0].collected_at.isoformat(),
        "latest_collected_at": snapshots[-1].collected_at.isoformat(),
        "timespan_days": times_days[-1],
        "doc_score_slope_per_month": float(doc_slope_month),
        "eng_maturity_slope_per_month": float(eng_slope_month),
        "archetype_stability": {
            "is_stable": is_stable,
            "history": archetypes,
            "initial_archetype": archetypes[0],
            "current_archetype": archetypes[-1],
        },
        "feature_deltas": feature_deltas,
        "timeline": snapshot_details,
    }


def run_growth_tests():
    """Run verification tests on real dataset users."""
    test_users = [
        "dineshkumarp07",
        "vijeth06",
        "Jeevashre12",
        "AMARNATH002",
        "RBROHANTH",
    ]

    print("\n" + "=" * 80)
    print("  DEV LENS - GROWTH TRAJECTORY SINGLE-SNAPSHOT VERIFICATION TEST")
    print("=" * 80)

    for username in test_users:
        res = get_growth_summary(username)
        print(f"\nDeveloper: {username}")
        print(f"  • Status:         {res.get('status')}")
        print(f"  • Snapshot Count: {res.get('snapshot_count')}")
        print(f"  • Message:        {res.get('message')}")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    run_growth_tests()

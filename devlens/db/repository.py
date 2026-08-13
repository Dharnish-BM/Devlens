"""
Repository layer for DevLens database operations.

All database reads and writes go through this module.
Never interact with session or models directly from business logic —
use these repository functions instead.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from devlens.db.models import (
    Developer, Snapshot, Feature,
    ClusterAssignment, ArchetypePrediction,
    DocScore, EngMaturityScore, CollectionExclusion,
    DocumentationScore, EngineeringMaturityScore,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Developer CRUD
# ---------------------------------------------------------------------------

def upsert_developer(
    session: Session,
    username: str,
    resume_source: Optional[str] = None,
) -> Developer:
    """Insert a new developer or update last_updated_at if they already exist.
    
    Returns the Developer ORM instance (either existing or newly created).
    """
    developer = session.query(Developer).filter_by(username=username).first()
    now = datetime.utcnow()

    if developer is None:
        developer = Developer(
            username=username,
            resume_source=resume_source,
            first_collected_at=now,
            last_updated_at=now,
        )
        session.add(developer)
        session.flush()  # flush to get the assigned id
        logger.info(f"Inserted new developer: '{username}' (id={developer.id})")
    else:
        developer.last_updated_at = now
        if resume_source is not None:
            developer.resume_source = resume_source
        session.flush()
        logger.info(f"Updated existing developer: '{username}' (id={developer.id})")

    return developer


def get_developer(session: Session, username: str) -> Optional[Developer]:
    """Retrieve a developer by username, or None if not found."""
    return session.query(Developer).filter_by(username=username).first()


# ---------------------------------------------------------------------------
# Snapshot CRUD
# ---------------------------------------------------------------------------

def insert_snapshot(
    session: Session,
    developer_id: int,
    raw_json_path: Optional[str] = None,
    collected_at: Optional[datetime] = None,
) -> Snapshot:
    """Create a new snapshot row for a developer collection run.
    
    Returns the newly created Snapshot instance.
    """
    snapshot = Snapshot(
        developer_id=developer_id,
        collected_at=collected_at or datetime.utcnow(),
        raw_json_path=raw_json_path,
    )
    session.add(snapshot)
    session.flush()  # flush to get the assigned id
    logger.info(f"Inserted snapshot id={snapshot.id} for developer_id={developer_id}")
    return snapshot


def get_latest_snapshot(session: Session, developer_id: int) -> Optional[Snapshot]:
    """Return the most recent snapshot for a developer."""
    return (
        session.query(Snapshot)
        .filter_by(developer_id=developer_id)
        .order_by(Snapshot.collected_at.desc())
        .first()
    )


def get_snapshot_history(session: Session, username: str) -> List[Dict[str, Any]]:
    """Return the full snapshot history for a developer by username.
    
    Used for growth trajectory queries. Returns a list of dicts
    (not ORM objects) for safe use outside the session context.
    """
    developer = get_developer(session, username)
    if developer is None:
        logger.warning(f"get_snapshot_history: developer '{username}' not found.")
        return []

    snapshots = (
        session.query(Snapshot)
        .filter_by(developer_id=developer.id)
        .order_by(Snapshot.collected_at.asc())
        .all()
    )

    history = []
    for snap in snapshots:
        feature_count = session.query(Feature).filter_by(snapshot_id=snap.id).count()
        history.append({
            "snapshot_id": snap.id,
            "developer_id": snap.developer_id,
            "collected_at": snap.collected_at.isoformat(),
            "raw_json_path": snap.raw_json_path,
            "feature_count": feature_count,
        })

    logger.info(f"Retrieved {len(history)} snapshots for developer '{username}'.")
    return history


# ---------------------------------------------------------------------------
# Feature bulk insert
# ---------------------------------------------------------------------------

def insert_features(
    session: Session,
    snapshot_id: int,
    features: Dict[str, float],
) -> int:
    """Bulk-insert features for a snapshot in long format.
    
    Args:
        session: Active SQLAlchemy session.
        snapshot_id: The snapshot to attach features to.
        features: Dict of {feature_name: feature_value}.
    
    Returns:
        Number of features inserted.
    """
    if not features:
        return 0

    feature_objs = [
        Feature(
            snapshot_id=snapshot_id,
            feature_name=name,
            feature_value=float(value) if value is not None else None,
        )
        for name, value in features.items()
    ]
    session.add_all(feature_objs)
    session.flush()
    logger.info(f"Inserted {len(feature_objs)} features for snapshot_id={snapshot_id}")
    return len(feature_objs)


def get_features(session: Session, snapshot_id: int) -> Dict[str, float]:
    """Retrieve all features for a snapshot as a flat dict."""
    rows = session.query(Feature).filter_by(snapshot_id=snapshot_id).all()
    return {row.feature_name: row.feature_value for row in rows}


# ---------------------------------------------------------------------------
# Score / Prediction CRUD
# ---------------------------------------------------------------------------

def upsert_cluster_assignment(
    session: Session,
    snapshot_id: int,
    cluster_id: int,
    distance_to_centroid: Optional[float] = None,
) -> ClusterAssignment:
    """Insert or replace cluster assignment for a snapshot."""
    existing = session.query(ClusterAssignment).filter_by(snapshot_id=snapshot_id).first()
    if existing:
        existing.cluster_id = cluster_id
        existing.distance_to_centroid = distance_to_centroid
        session.flush()
        return existing

    obj = ClusterAssignment(
        snapshot_id=snapshot_id,
        cluster_id=cluster_id,
        distance_to_centroid=distance_to_centroid,
    )
    session.add(obj)
    session.flush()
    return obj


def insert_archetype_prediction(
    session: Session,
    snapshot_id: int,
    archetype_label: str,
    confidence: float,
    shap_top_features: Optional[Dict[str, float]] = None,
) -> ArchetypePrediction:
    """Insert an archetype prediction for a snapshot."""
    obj = ArchetypePrediction(
        snapshot_id=snapshot_id,
        archetype_label=archetype_label,
        confidence=confidence,
        shap_top_features=shap_top_features,
    )
    session.add(obj)
    session.flush()
    return obj


def upsert_doc_score(
    session: Session,
    snapshot_id: int,
    score: float,
    components: Optional[Dict[str, Any]] = None,
) -> DocScore:
    """Insert or update documentation score for a snapshot.
    
    components JSON stores weight breakdown: desc_coverage, desc_depth, bio, blog, pinned.
    """
    existing = session.query(DocScore).filter_by(snapshot_id=snapshot_id).first()
    if existing:
        existing.score = score
        existing.components = components
        session.flush()
        return existing

    obj = DocScore(snapshot_id=snapshot_id, score=score, components=components)
    session.add(obj)
    session.flush()
    return obj


# Alias for backward compatibility
upsert_documentation_score = upsert_doc_score


def upsert_eng_maturity_score(
    session: Session,
    snapshot_id: int,
    score: float,
    components: Optional[Dict[str, Any]] = None,
) -> EngMaturityScore:
    """Insert or update engineering maturity score for a snapshot.
    
    components JSON stores weight breakdown: eng_ci_ratio, eng_test_ratio, eng_commit_message_quality, eng_pr_discipline, eng_review_participation.
    """
    existing = session.query(EngMaturityScore).filter_by(snapshot_id=snapshot_id).first()
    if existing:
        existing.score = score
        existing.components = components
        session.flush()
        return existing

    obj = EngMaturityScore(snapshot_id=snapshot_id, score=score, components=components)
    session.add(obj)
    session.flush()
    return obj


# Alias for backward compatibility
upsert_engineering_maturity_score = upsert_eng_maturity_score


# ---------------------------------------------------------------------------
# Collection Exclusion CRUD
# ---------------------------------------------------------------------------

def insert_exclusion(
    session: Session,
    username: str,
    reason: str,
    excluded_at: Optional[datetime] = None,
) -> CollectionExclusion:
    """Record a classmate/username dropped during collection (404, unparseable entry, etc.)."""
    exclusion = CollectionExclusion(
        username=username,
        reason=reason,
        excluded_at=excluded_at or datetime.utcnow(),
    )
    session.add(exclusion)
    session.flush()
    logger.info(f"Inserted collection exclusion for '{username}': {reason}")
    return exclusion


def get_all_exclusions(session: Session) -> List[CollectionExclusion]:
    """Retrieve all recorded collection exclusions."""
    return session.query(CollectionExclusion).order_by(CollectionExclusion.excluded_at.asc()).all()


# ---------------------------------------------------------------------------
# Current Features DataFrame Query for ML / Clustering
# ---------------------------------------------------------------------------

def get_all_current_features(session: Session) -> pd.DataFrame:
    """Retrieve all current features across developers for Phase 5 clustering.
    
    Identifies the latest snapshot per developer and pulls its features,
    returning a pandas DataFrame with one row per developer (indexed by username)
    and one column per feature.
    """
    subq = (
        session.query(
            Snapshot.developer_id,
            func.max(Snapshot.id).label("max_snapshot_id")
        )
        .group_by(Snapshot.developer_id)
        .subquery()
    )

    rows = (
        session.query(Developer.username, Feature.feature_name, Feature.feature_value)
        .select_from(Developer)
        .join(Snapshot, Developer.id == Snapshot.developer_id)
        .join(subq, Snapshot.id == subq.c.max_snapshot_id)
        .join(Feature, Feature.snapshot_id == Snapshot.id)
        .all()
    )

    if not rows:
        df = pd.DataFrame()
        df.index.name = "username"
        return df

    df = pd.DataFrame(rows, columns=["username", "feature_name", "feature_value"])
    pivoted = df.pivot(index="username", columns="feature_name", values="feature_value")
    pivoted.index.name = "username"
    pivoted.columns.name = None
    return pivoted


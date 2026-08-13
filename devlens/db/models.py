"""
SQLAlchemy ORM models for the DevLens database.

All tables are defined here and imported by session.py for table creation.
Long-format 'features' table allows new ML features without schema migrations.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey,
    Text, UniqueConstraint, JSON, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Developer(Base):
    """One row per GitHub username ever profiled."""
    __tablename__ = "developers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    resume_source = Column(String(512), nullable=True)  # path/filename of resume if discovered via file
    first_collected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    snapshots = relationship("Snapshot", back_populates="developer", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Developer(id={self.id}, username='{self.username}')>"


class Snapshot(Base):
    """One row per collection run for a developer — enables growth trajectory tracking."""
    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    developer_id = Column(Integer, ForeignKey("developers.id", ondelete="CASCADE"), nullable=False)
    collected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    raw_json_path = Column(String(512), nullable=True)  # path to data/raw/{username}.json

    developer = relationship("Developer", back_populates="snapshots")
    features = relationship("Feature", back_populates="snapshot", cascade="all, delete-orphan")
    cluster_assignment = relationship("ClusterAssignment", back_populates="snapshot", uselist=False, cascade="all, delete-orphan")
    archetype_predictions = relationship("ArchetypePrediction", back_populates="snapshot", cascade="all, delete-orphan")
    doc_score = relationship("DocScore", back_populates="snapshot", uselist=False, cascade="all, delete-orphan")
    eng_maturity_score = relationship("EngMaturityScore", back_populates="snapshot", uselist=False, cascade="all, delete-orphan")

    @property
    def documentation_score(self):
        return self.doc_score

    @property
    def engineering_maturity_score(self):
        return self.eng_maturity_score

    __table_args__ = (
        Index("ix_snapshots_developer_collected", "developer_id", "collected_at"),
    )

    def __repr__(self) -> str:
        return f"<Snapshot(id={self.id}, developer_id={self.developer_id}, collected_at={self.collected_at})>"


class Feature(Base):
    """Long-format feature store — one row per (snapshot, feature_name).
    
    Long format means new ML features can be added to any snapshot without
    requiring schema migrations or NULL-heavy wide tables.
    """
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey("snapshots.id", ondelete="CASCADE"), nullable=False)
    feature_name = Column(String(255), nullable=False)
    feature_value = Column(Float, nullable=True)

    snapshot = relationship("Snapshot", back_populates="features")

    __table_args__ = (
        UniqueConstraint("snapshot_id", "feature_name", name="uq_features_snapshot_name"),
        Index("ix_features_snapshot_id", "snapshot_id"),
        Index("ix_features_feature_name", "feature_name"),
    )

    def __repr__(self) -> str:
        return f"<Feature(snapshot_id={self.snapshot_id}, name='{self.feature_name}', value={self.feature_value})>"


class ClusterAssignment(Base):
    """Cluster assignment result for a snapshot (one per snapshot)."""
    __tablename__ = "cluster_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey("snapshots.id", ondelete="CASCADE"), nullable=False, unique=True)
    cluster_id = Column(Integer, nullable=False)
    distance_to_centroid = Column(Float, nullable=True)

    snapshot = relationship("Snapshot", back_populates="cluster_assignment")

    def __repr__(self) -> str:
        return f"<ClusterAssignment(snapshot_id={self.snapshot_id}, cluster_id={self.cluster_id})>"


class ArchetypePrediction(Base):
    """Archetype classification result for a snapshot (multiple archetypes possible with confidence scores)."""
    __tablename__ = "archetype_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey("snapshots.id", ondelete="CASCADE"), nullable=False)
    archetype_label = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    shap_top_features = Column(JSON, nullable=True)  # dict of {feature_name: shap_value}

    snapshot = relationship("Snapshot", back_populates="archetype_predictions")

    __table_args__ = (
        Index("ix_archetype_snapshot_id", "snapshot_id"),
    )

    def __repr__(self) -> str:
        return f"<ArchetypePrediction(snapshot_id={self.snapshot_id}, label='{self.archetype_label}', confidence={self.confidence})>"


class DocScore(Base):
    """Computed documentation quality score for a snapshot."""
    __tablename__ = "doc_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey("snapshots.id", ondelete="CASCADE"), nullable=False, unique=True)
    score = Column(Float, nullable=False)
    components = Column(JSON, nullable=True)  # stores breakdown: desc_coverage, desc_depth, bio, blog, pinned

    snapshot = relationship("Snapshot", back_populates="doc_score")

    def __repr__(self) -> str:
        return f"<DocScore(snapshot_id={self.snapshot_id}, score={self.score})>"


class EngMaturityScore(Base):
    """Computed engineering maturity score for a snapshot."""
    __tablename__ = "eng_maturity_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey("snapshots.id", ondelete="CASCADE"), nullable=False, unique=True)
    score = Column(Float, nullable=False)
    components = Column(JSON, nullable=True)  # stores breakdown: eng_ci_ratio, eng_test_ratio, eng_commit_message_quality, eng_pr_discipline, eng_review_participation

    snapshot = relationship("Snapshot", back_populates="eng_maturity_score")

    def __repr__(self) -> str:
        return f"<EngMaturityScore(snapshot_id={self.snapshot_id}, score={self.score})>"


class CollectionExclusion(Base):
    """Records classmates/usernames dropped during collection (404s, unparseable entries, etc.)."""
    __tablename__ = "collection_exclusions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), nullable=False, index=True)
    reason = Column(Text, nullable=False)
    excluded_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<CollectionExclusion(username='{self.username}', reason='{self.reason}')>"


# Backwards compatibility aliases
DocumentationScore = DocScore
EngineeringMaturityScore = EngMaturityScore


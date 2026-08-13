"""
Tests for the DevLens database layer.

Uses in-memory SQLite to avoid any on-disk side effects.
Covers: upsert_developer, insert_snapshot, insert_features (bulk),
get_snapshot_history, score upserts, and archetype predictions.
"""

import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from devlens.db.models import Base
from devlens.db.session import init_db
from devlens.db import repository as repo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def engine():
    """Fresh in-memory SQLite engine per test function."""
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    init_db(eng)
    yield eng
    eng.dispose()


@pytest.fixture(scope="function")
def session(engine):
    """Active session wired to the in-memory engine, auto-committed after test."""
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    sess = Session()
    yield sess
    sess.rollback()
    sess.close()


# ---------------------------------------------------------------------------
# Developer tests
# ---------------------------------------------------------------------------

class TestDeveloper:
    def test_upsert_creates_new_developer(self, session):
        dev = repo.upsert_developer(session, username="octocat", resume_source="resume.pdf")
        session.commit()

        assert dev.id is not None
        assert dev.username == "octocat"
        assert dev.resume_source == "resume.pdf"
        assert dev.first_collected_at is not None

    def test_upsert_existing_updates_timestamp(self, session):
        dev1 = repo.upsert_developer(session, username="octocat")
        session.commit()
        original_first = dev1.first_collected_at

        # Upsert again — should NOT change first_collected_at
        dev2 = repo.upsert_developer(session, username="octocat", resume_source="updated.pdf")
        session.commit()

        assert dev2.id == dev1.id
        assert dev2.first_collected_at == original_first
        assert dev2.resume_source == "updated.pdf"

    def test_get_developer_returns_none_for_unknown(self, session):
        result = repo.get_developer(session, "no_such_user")
        assert result is None

    def test_get_developer_returns_existing(self, session):
        repo.upsert_developer(session, username="octocat")
        session.commit()

        result = repo.get_developer(session, "octocat")
        assert result is not None
        assert result.username == "octocat"


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_insert_snapshot_creates_row(self, session):
        dev = repo.upsert_developer(session, username="octocat")
        session.commit()

        snap = repo.insert_snapshot(session, developer_id=dev.id, raw_json_path="data/raw/octocat.json")
        session.commit()

        assert snap.id is not None
        assert snap.developer_id == dev.id
        assert snap.raw_json_path == "data/raw/octocat.json"

    def test_get_latest_snapshot(self, session):
        dev = repo.upsert_developer(session, username="octocat")
        session.commit()

        t1 = datetime(2025, 1, 1)
        t2 = datetime(2025, 6, 1)
        snap1 = repo.insert_snapshot(session, developer_id=dev.id, collected_at=t1)
        snap2 = repo.insert_snapshot(session, developer_id=dev.id, collected_at=t2)
        session.commit()

        latest = repo.get_latest_snapshot(session, developer_id=dev.id)
        assert latest.id == snap2.id

    def test_get_snapshot_history_ordered_asc(self, session):
        dev = repo.upsert_developer(session, username="octocat")
        session.commit()

        for i in range(3):
            repo.insert_snapshot(session, developer_id=dev.id, collected_at=datetime(2025, i + 1, 1))
        session.commit()

        history = repo.get_snapshot_history(session, "octocat")
        assert len(history) == 3
        timestamps = [h["collected_at"] for h in history]
        assert timestamps == sorted(timestamps)

    def test_get_snapshot_history_unknown_user(self, session):
        result = repo.get_snapshot_history(session, "ghost_user_xyz")
        assert result == []


# ---------------------------------------------------------------------------
# Feature tests
# ---------------------------------------------------------------------------

class TestFeatures:
    def _make_dev_and_snap(self, session):
        dev = repo.upsert_developer(session, username="featuretest")
        session.commit()
        snap = repo.insert_snapshot(session, developer_id=dev.id)
        session.commit()
        return dev, snap

    def test_insert_features_bulk(self, session):
        _, snap = self._make_dev_and_snap(session)
        features = {
            "total_commits": 3219,
            "repo_count": 12,
            "avg_stars": 124.5,
            "primary_language_c": 1.0,
        }
        count = repo.insert_features(session, snap.id, features)
        session.commit()

        assert count == 4

    def test_get_features_round_trip(self, session):
        _, snap = self._make_dev_and_snap(session)
        features = {"total_commits": 100.0, "repo_count": 5.0}
        repo.insert_features(session, snap.id, features)
        session.commit()

        retrieved = repo.get_features(session, snap.id)
        assert retrieved["total_commits"] == 100.0
        assert retrieved["repo_count"] == 5.0

    def test_features_reflected_in_snapshot_history(self, session):
        dev, snap = self._make_dev_and_snap(session)
        repo.insert_features(session, snap.id, {"commits": 50.0, "stars": 10.0})
        session.commit()

        history = repo.get_snapshot_history(session, "featuretest")
        assert history[0]["feature_count"] == 2


# ---------------------------------------------------------------------------
# Score & Prediction tests
# ---------------------------------------------------------------------------

class TestScoresAndPredictions:
    def _make_snap(self, session, username="scoretest"):
        dev = repo.upsert_developer(session, username=username)
        session.commit()
        snap = repo.insert_snapshot(session, developer_id=dev.id)
        session.commit()
        return snap

    def test_upsert_cluster_assignment(self, session):
        snap = self._make_snap(session)
        ca = repo.upsert_cluster_assignment(session, snap.id, cluster_id=2, distance_to_centroid=0.34)
        session.commit()

        assert ca.cluster_id == 2
        assert ca.distance_to_centroid == pytest.approx(0.34)

    def test_cluster_assignment_upsert_updates(self, session):
        snap = self._make_snap(session)
        repo.upsert_cluster_assignment(session, snap.id, cluster_id=1, distance_to_centroid=1.0)
        session.commit()
        repo.upsert_cluster_assignment(session, snap.id, cluster_id=3, distance_to_centroid=0.1)
        session.commit()

        from devlens.db.models import ClusterAssignment
        rows = session.query(ClusterAssignment).filter_by(snapshot_id=snap.id).all()
        assert len(rows) == 1
        assert rows[0].cluster_id == 3

    def test_insert_archetype_prediction_with_shap(self, session):
        snap = self._make_snap(session)
        shap = {"total_commits": 0.42, "repo_count": 0.18}
        pred = repo.insert_archetype_prediction(session, snap.id, "systems_engineer", 0.87, shap)
        session.commit()

        assert pred.archetype_label == "systems_engineer"
        assert pred.confidence == pytest.approx(0.87)
        assert pred.shap_top_features["total_commits"] == pytest.approx(0.42)

    def test_upsert_doc_score_exact_handoff_components(self, session):
        snap = self._make_snap(session)
        components = {
            "desc_coverage": 1.0,
            "desc_depth": 0.6514,
            "bio": 0.0,
            "blog": 0.0,
            "pinned": 0.8333,
        }
        score = repo.upsert_doc_score(session, snap.id, score=0.6295, components=components)
        session.commit()

        assert score.score == pytest.approx(0.6295)
        assert score.components["desc_coverage"] == 1.0
        assert score.components["desc_depth"] == pytest.approx(0.6514)

    def test_upsert_eng_maturity_score_exact_handoff_components(self, session):
        snap = self._make_snap(session)
        components = {
            "eng_ci_ratio": 0.1111,
            "eng_test_ratio": 0.1111,
            "eng_commit_message_quality": 0.1872,
            "eng_pr_discipline": 0.8471,
            "eng_review_participation": 1.0,
        }
        score = repo.upsert_eng_maturity_score(session, snap.id, score=0.3256, components=components)
        session.commit()

        assert score.score == pytest.approx(0.3256)
        assert score.components["eng_ci_ratio"] == pytest.approx(0.1111)
        assert score.components["eng_review_participation"] == 1.0


# ---------------------------------------------------------------------------
# Collection Exclusion tests
# ---------------------------------------------------------------------------

class TestExclusions:
    def test_insert_and_get_exclusions(self, session):
        ex1 = repo.insert_exclusion(session, username="anbuchelvan-efx", reason="Could not retrieve user info (404)")
        ex2 = repo.insert_exclusion(session, username="Abisha_Rebekkal", reason="Invalid username format")
        session.commit()

        exclusions = repo.get_all_exclusions(session)
        assert len(exclusions) == 2
        usernames = [e.username for e in exclusions]
        assert "anbuchelvan-efx" in usernames
        assert "Abisha_Rebekkal" in usernames


# ---------------------------------------------------------------------------
# DataFrame / Clustering Feature Matrix tests
# ---------------------------------------------------------------------------

class TestCurrentFeaturesDataFrame:
    def test_get_all_current_features_df(self, session):
        # Developer 1: octocat (2 snapshots, should use snapshot 2)
        dev1 = repo.upsert_developer(session, username="octocat")
        session.commit()
        snap1_old = repo.insert_snapshot(session, developer_id=dev1.id, collected_at=datetime(2025, 1, 1))
        snap1_new = repo.insert_snapshot(session, developer_id=dev1.id, collected_at=datetime(2025, 6, 1))
        session.commit()
        repo.insert_features(session, snap1_old.id, {"commits": 10.0, "stars": 5.0})
        repo.insert_features(session, snap1_new.id, {"commits": 50.0, "stars": 20.0})
        session.commit()

        # Developer 2: torvalds (1 snapshot)
        dev2 = repo.upsert_developer(session, username="torvalds")
        session.commit()
        snap2 = repo.insert_snapshot(session, developer_id=dev2.id, collected_at=datetime(2025, 6, 1))
        session.commit()
        repo.insert_features(session, snap2.id, {"commits": 3000.0, "stars": 100000.0})
        session.commit()

        df = repo.get_all_current_features(session)

        # Check structure
        assert df.shape == (2, 2)
        assert set(df.index) == {"octocat", "torvalds"}
        assert set(df.columns) == {"commits", "stars"}

        # Verify values resolve to latest snapshot
        assert df.loc["octocat", "commits"] == 50.0
        assert df.loc["octocat", "stars"] == 20.0
        assert df.loc["torvalds", "commits"] == 3000.0


# ---------------------------------------------------------------------------
# Full integration: developer + snapshot + features round-trip
# ---------------------------------------------------------------------------

class TestFullIntegration:
    def test_full_write_and_read_back(self, session):
        """Insert a developer, snapshot, features, cluster, archetype, doc/maturity scores — read them all back."""
        # Write
        dev = repo.upsert_developer(session, username="torvalds", resume_source="resume_direct_url.pdf")
        session.commit()

        snap = repo.insert_snapshot(session, developer_id=dev.id, raw_json_path="data/raw/torvalds.json")
        session.commit()

        features = {
            "total_commits": 3219.0,
            "repo_count": 12.0,
            "avg_stars": 20145.4,
            "primary_language_c": 1.0,
            "fork_ratio": 0.17,
        }
        count = repo.insert_features(session, snap.id, features)
        session.commit()

        repo.upsert_cluster_assignment(session, snap.id, cluster_id=0, distance_to_centroid=0.12)
        repo.insert_archetype_prediction(session, snap.id, "systems_engineer", 0.95,
                                         {"total_commits": 0.55, "primary_language_c": 0.31})
        repo.upsert_doc_score(session, snap.id, 0.6295, {
            "desc_coverage": 1.0, "desc_depth": 0.6514, "bio": 0.0, "blog": 0.0, "pinned": 0.8333
        })
        repo.upsert_eng_maturity_score(session, snap.id, 0.3256, {
            "eng_ci_ratio": 0.1111, "eng_test_ratio": 0.1111, "eng_commit_message_quality": 0.1872,
            "eng_pr_discipline": 0.8471, "eng_review_participation": 1.0
        })
        session.commit()

        # Read back
        history = repo.get_snapshot_history(session, "torvalds")
        assert len(history) == 1
        assert history[0]["feature_count"] == 5
        assert history[0]["raw_json_path"] == "data/raw/torvalds.json"

        retrieved_features = repo.get_features(session, snap.id)
        assert retrieved_features["total_commits"] == pytest.approx(3219.0)
        assert retrieved_features["repo_count"] == pytest.approx(12.0)

        developer = repo.get_developer(session, "torvalds")
        assert developer.resume_source == "resume_direct_url.pdf"
        assert len(developer.snapshots) == 1
        assert len(developer.snapshots[0].features) == 5
        assert developer.snapshots[0].doc_score.score == pytest.approx(0.6295)
        assert developer.snapshots[0].eng_maturity_score.score == pytest.approx(0.3256)


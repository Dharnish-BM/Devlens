"""
Tests for the DevLens database layer in tests/test_db.py.

Uses in-memory SQLite to avoid any on-disk side effects.
Covers: upsert_developer, insert_snapshot, insert_features (bulk),
get_snapshot_history, score upserts, exclusions, current features DataFrame,
and full integration.
"""

import sys
import os
from datetime import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Ensure project root is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from devlens.db.models import Base
from devlens.db.session import init_db
from devlens.db import repository as repo


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

        dev2 = repo.upsert_developer(session, username="octocat", resume_source="updated.pdf")
        session.commit()

        assert dev2.id == dev1.id
        assert dev2.first_collected_at == original_first
        assert dev2.resume_source == "updated.pdf"


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


class TestFeatures:
    def test_insert_and_get_features(self, session):
        dev = repo.upsert_developer(session, username="featuretest")
        session.commit()
        snap = repo.insert_snapshot(session, developer_id=dev.id)
        session.commit()

        features = {"total_commits": 3219.0, "repo_count": 12.0, "avg_stars": 124.5}
        count = repo.insert_features(session, snap.id, features)
        session.commit()

        assert count == 3
        retrieved = repo.get_features(session, snap.id)
        assert retrieved["total_commits"] == 3219.0
        assert retrieved["repo_count"] == 12.0


class TestExclusions:
    def test_insert_and_get_exclusions(self, session):
        repo.insert_exclusion(session, username="anbuchelvan-efx", reason="Could not retrieve user info (404)")
        repo.insert_exclusion(session, username="Abisha_Rebekkal", reason="Invalid username format")
        session.commit()

        exclusions = repo.get_all_exclusions(session)
        assert len(exclusions) == 2
        usernames = [e.username for e in exclusions]
        assert "anbuchelvan-efx" in usernames
        assert "Abisha_Rebekkal" in usernames


class TestCurrentFeaturesDataFrame:
    def test_get_all_current_features_df(self, session):
        dev1 = repo.upsert_developer(session, username="octocat")
        session.commit()
        snap1_old = repo.insert_snapshot(session, developer_id=dev1.id, collected_at=datetime(2025, 1, 1))
        snap1_new = repo.insert_snapshot(session, developer_id=dev1.id, collected_at=datetime(2025, 6, 1))
        session.commit()
        repo.insert_features(session, snap1_old.id, {"commits": 10.0, "stars": 5.0})
        repo.insert_features(session, snap1_new.id, {"commits": 50.0, "stars": 20.0})
        session.commit()

        dev2 = repo.upsert_developer(session, username="torvalds")
        session.commit()
        snap2 = repo.insert_snapshot(session, developer_id=dev2.id, collected_at=datetime(2025, 6, 1))
        session.commit()
        repo.insert_features(session, snap2.id, {"commits": 3000.0, "stars": 100000.0})
        session.commit()

        df = repo.get_all_current_features(session)

        assert df.shape == (2, 2)
        assert set(df.index) == {"octocat", "torvalds"}
        assert df.loc["octocat", "commits"] == 50.0
        assert df.loc["octocat", "stars"] == 20.0
        assert df.loc["torvalds", "commits"] == 3000.0

    def test_prepare_scaled_feature_matrix_drops_zero_variance(self, session):
        dev1 = repo.upsert_developer(session, username="user1")
        dev2 = repo.upsert_developer(session, username="user2")
        session.commit()
        snap1 = repo.insert_snapshot(session, developer_id=dev1.id)
        snap2 = repo.insert_snapshot(session, developer_id=dev2.id)
        session.commit()

        # "go_lang" has zero variance (0.0 for both users), "commits" varies
        repo.insert_features(session, snap1.id, {"commits": 10.0, "go_lang": 0.0})
        repo.insert_features(session, snap2.id, {"commits": 50.0, "go_lang": 0.0})
        session.commit()

        from devlens.features.feature_engineering import prepare_scaled_feature_matrix
        df = repo.get_all_current_features(session)

        scaled_df, scaler, dropped = prepare_scaled_feature_matrix(df, drop_zero_variance=True)

        assert dropped == ["go_lang"]
        assert "go_lang" not in scaled_df.columns
        assert "commits" in scaled_df.columns
        assert scaled_df.isna().sum().sum() == 0
        assert (scaled_df.values != scaled_df.values).sum() == 0  # no NaNs


class TestFullIntegration:
    def test_full_write_and_read_back(self, session):
        """Fake developer + snapshot + features + scores read back test."""
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

        # Read back verification
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

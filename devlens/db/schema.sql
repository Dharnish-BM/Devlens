-- DevLens Schema SQL Reference
-- This file is the canonical schema definition.
-- The live database is managed via SQLAlchemy ORM (models.py).
-- Use `python -m devlens.db.init_db` to initialize the database.

CREATE TABLE IF NOT EXISTS developers (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    username            TEXT    NOT NULL UNIQUE,
    resume_source       TEXT,
    first_collected_at  DATETIME NOT NULL DEFAULT (datetime('now')),
    last_updated_at     DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_developers_username ON developers (username);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    developer_id    INTEGER  NOT NULL REFERENCES developers(id) ON DELETE CASCADE,
    collected_at    DATETIME NOT NULL DEFAULT (datetime('now')),
    raw_json_path   TEXT
);

CREATE INDEX IF NOT EXISTS ix_snapshots_developer_collected
    ON snapshots (developer_id, collected_at);

-- ---------------------------------------------------------------------------
-- Long-format feature store. One row per (snapshot, feature_name).
-- Adding new ML features never requires schema changes.

CREATE TABLE IF NOT EXISTS features (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id     INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    feature_name    TEXT    NOT NULL,
    feature_value   REAL,
    UNIQUE (snapshot_id, feature_name)
);

CREATE INDEX IF NOT EXISTS ix_features_snapshot_id  ON features (snapshot_id);
CREATE INDEX IF NOT EXISTS ix_features_feature_name ON features (feature_name);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS cluster_assignments (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id             INTEGER NOT NULL UNIQUE REFERENCES snapshots(id) ON DELETE CASCADE,
    cluster_id              INTEGER NOT NULL,
    distance_to_centroid    REAL
);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS archetype_predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id         INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
    archetype_label     TEXT    NOT NULL,
    confidence          REAL    NOT NULL,
    shap_top_features   TEXT    -- stored as JSON blob
);

CREATE INDEX IF NOT EXISTS ix_archetype_snapshot_id ON archetype_predictions (snapshot_id);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS doc_scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL UNIQUE REFERENCES snapshots(id) ON DELETE CASCADE,
    score       REAL    NOT NULL,
    components  TEXT    -- stored as JSON blob: desc_coverage, desc_depth, bio, blog, pinned
);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS eng_maturity_scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL UNIQUE REFERENCES snapshots(id) ON DELETE CASCADE,
    score       REAL    NOT NULL,
    components  TEXT    -- stored as JSON blob: eng_ci_ratio, eng_test_ratio, eng_commit_message_quality, eng_pr_discipline, eng_review_participation
);

-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS collection_exclusions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL,
    reason      TEXT    NOT NULL,
    excluded_at DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_collection_exclusions_username ON collection_exclusions (username);


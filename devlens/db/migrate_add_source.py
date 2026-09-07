"""
Migration script: Add 'source' column to 'developers' table and backfill existing rows.

Values:
- 'consented_cohort': The fixed, empirical 188-student research cohort.
- 'live_upload': New candidates collected via live/resume pipeline going forward.
"""

import logging
import sqlite3
from pathlib import Path
from sqlalchemy import text
from devlens.db.session import get_engine, get_session
from devlens.db.models import Developer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def migrate():
    engine = get_engine()
    
    with engine.connect() as conn:
        # 1. Check existing columns in developers table
        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(developers)")).fetchall()]
        logger.info(f"Existing columns in developers table: {columns}")
        
        if "source" not in columns:
            logger.info("Adding 'source' column to 'developers' table...")
            conn.execute(text("ALTER TABLE developers ADD COLUMN source VARCHAR(50) DEFAULT 'live_upload' NOT NULL"))
            conn.commit()
            logger.info("Column 'source' added successfully.")
        else:
            logger.info("Column 'source' already exists.")
            
        # 2. Backfill all existing developers to 'consented_cohort'
        logger.info("Backfilling existing developers to 'consented_cohort'...")
        result = conn.execute(text("UPDATE developers SET source = 'consented_cohort' WHERE source IS NULL OR source = 'live_upload'"))
        conn.commit()
        logger.info(f"Updated rows: {result.rowcount}")
        
        # 3. Add index on source if not present
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_developers_source ON developers (source)"))
            conn.commit()
        except Exception as e:
            logger.warning(f"Index creation notice: {e}")

    # 4. Verify counts using ORM
    with get_session() as session:
        total = session.query(Developer).count()
        consented = session.query(Developer).filter_by(source="consented_cohort").count()
        live = session.query(Developer).filter_by(source="live_upload").count()
        
        print("\n" + "=" * 80)
        print("  DEV LENS - DEVELOPER SOURCE COLUMN MIGRATION & BACKFILL REPORT")
        print("=" * 80)
        print(f"Total Developers in DB:               {total}")
        print(f"Tagged 'consented_cohort' (Research): {consented}")
        print(f"Tagged 'live_upload' (Production):    {live}")
        print("=" * 80)
        
        assert total == 188, f"Expected 188 developers, found {total}"
        assert consented == 188, f"Expected 188 consented_cohort developers, found {consented}"
        print(" [SUCCESS] All 188 existing developers are strictly tagged as 'consented_cohort'!\n")


if __name__ == "__main__":
    migrate()

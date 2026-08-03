"""
DevLens database initialisation script.

Creates all tables defined in models.py in the configured SQLite database.
Safe to run multiple times — uses CREATE TABLE IF NOT EXISTS semantics.

Usage:
    python -m devlens.db.init_db
    python -m devlens.db.init_db --db-path custom/path/to/devlens.db
    python -m devlens.db.init_db --echo
"""

import argparse
import logging

from sqlalchemy import inspect

from devlens.db.session import get_engine, init_db
from devlens.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_init(db_path: str = None, echo: bool = False) -> None:
    db_url = f"sqlite:///{db_path}" if db_path else None
    engine = get_engine(db_url=db_url or "sqlite:///devlens.db", echo=echo)
    init_db(engine)

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print("\n" + "=" * 50)
    print("  DEV LENS - DATABASE INITIALISED")
    print("=" * 50)
    print(f" Database URL  : {engine.url}")
    print(f" Tables created: {len(tables)}")
    for table in sorted(tables):
        cols = [col["name"] for col in inspector.get_columns(table)]
        print(f"   • {table:<35} ({len(cols)} columns)")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Initialise the DevLens SQLite database.")
    parser.add_argument("--db-path", type=str, default=None, help="Override SQLite DB file path.")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements to stdout.")
    args = parser.parse_args()
    run_init(db_path=args.db_path, echo=args.echo)


if __name__ == "__main__":
    main()

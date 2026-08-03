"""
SQLAlchemy engine and session factory for DevLens.

Reads DB_PATH from .env (defaulting to devlens.db in project root).
Use get_session() as a context manager for all database access.
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from devlens.db.models import Base

load_dotenv()

# DB path resolution — use env override or default to project root devlens.db
_DB_PATH = os.getenv("DEVLENS_DB_PATH", "devlens.db")
_DATABASE_URL = f"sqlite:///{_DB_PATH}"


def get_engine(db_url: str = _DATABASE_URL, echo: bool = False) -> Engine:
    """Create and return a SQLAlchemy engine.
    
    Args:
        db_url: SQLAlchemy connection URL. Defaults to SQLite at devlens.db.
        echo: If True, log all SQL statements (useful for debugging).
    """
    engine = create_engine(
        db_url,
        echo=echo,
        connect_args={"check_same_thread": False},  # required for SQLite in multi-threaded use
    )
    return engine


def init_db(engine: Engine = None) -> Engine:
    """Create all tables if they don't already exist.
    
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS semantics.
    """
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


# Module-level default engine and session factory
_engine: Engine = None
_SessionFactory: sessionmaker = None


def _get_default_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = init_db()
    return _engine


def _get_session_factory() -> sessionmaker:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=_get_default_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def get_session(engine: Engine = None) -> Generator[Session, None, None]:
    """Context manager yielding a database session with automatic commit/rollback.
    
    Usage:
        with get_session() as session:
            session.add(some_model)
    """
    factory = sessionmaker(bind=engine or _get_default_engine(), expire_on_commit=False)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

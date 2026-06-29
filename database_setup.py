"""
Peptibro Database Setup
Works both locally and on Streamlit Community Cloud.
Uses SQLite (file-based, no external DB needed).
"""

import sqlite3
from pathlib import Path
import streamlit as st

# On Streamlit Cloud, use /tmp for writable storage
# Locally, use ./db/
def _get_db_path() -> Path:
    """Get database path based on environment."""
    import os
    if os.getenv("STREAMLIT_CLOUD") or Path("/tmp").exists():
        # Streamlit Cloud - use /tmp (writable but ephemeral)
        return Path("/tmp") / "peptibro.db"
    # Local development
    return Path("db") / "peptibro.db"

DB_PATH = _get_db_path()


@st.cache_resource
def get_connection():
    """Get SQLite connection (cached for performance)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            compound_name TEXT NOT NULL,
            dosage_mcg REAL NOT NULL,
            notes TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS blood_markers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            test_name TEXT,
            igf1 REAL,
            glucose REAL,
            free_testosterone REAL,
            total_testosterone REAL,
            estradiol REAL,
            cholesterol_total REAL,
            ldl REAL,
            hdl REAL,
            triglycerides REAL,
            alt REAL,
            ast REAL,
            tsh REAL,
            creatinine REAL,
            notes TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oracle_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            source TEXT,
            model_used TEXT
        )
    """)

    conn.commit()
    print(f"[Peptibro] Database initialized at {DB_PATH}")


if __name__ == "__main__":
    init_database()

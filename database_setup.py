"""
Peptibro Database Setup - Robust Version for Streamlit Cloud
"""

import sqlite3
import os
from pathlib import Path
import streamlit as st


def _get_db_path():
    """Get database path."""
    if os.environ.get("STREAMLIT_CLOUD"):
        return "/tmp/peptibro.db"
    db_dir = Path(__file__).parent / "db"
    db_dir.mkdir(exist_ok=True)
    return str(db_dir / "peptibro.db")


DB_PATH = _get_db_path()


def _create_tables(conn):
    """Create all tables if they don't exist."""
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
            igf1 REAL, glucose REAL, free_testosterone REAL,
            total_testosterone REAL, estradiol REAL,
            cholesterol_total REAL, ldl REAL, hdl REAL,
            triglycerides REAL, alt REAL, ast REAL,
            tsh REAL, creatinine REAL, notes TEXT
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


@st.cache_resource
def get_connection():
    """Get cached SQLite connection."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _create_tables(conn)
        return conn
    except Exception as e:
        st.error(f"DB Error: {e}")
        return None


def init_database():
    """Initialize database."""
    conn = get_connection()
    if conn:
        _create_tables(conn)
    return conn is not None


def read_sql(query, conn=None):
    """Safe SQL read with error handling."""
    try:
        if conn is None:
            conn = get_connection()
        if conn is None:
            return __import__("pandas").DataFrame()
        return __import__("pandas").read_sql(query, conn)
    except Exception:
        return __import__("pandas").DataFrame()


def execute_write(query, params=None):
    """Safe SQL write with error handling."""
    try:
        conn = get_connection()
        if conn is None:
            return False
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        conn.commit()
        return True
    except Exception:
        return False


# Initialize on import
init_database()

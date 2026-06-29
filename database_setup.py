"""
Peptibro Database Setup - Simple In-Memory Version
Uses only session state. No SQLite dependency.
"""

import streamlit as st
import pandas as pd
from datetime import datetime


def init_database():
    """Initialize in-memory storage."""
    if "db_initialized" not in st.session_state:
        st.session_state.daily_log = []
        st.session_state.blood_markers = []
        st.session_state.oracle_history = []
        st.session_state.next_id = {"daily_log": 1, "blood_markers": 1, "oracle_history": 1}
        st.session_state.db_initialized = True
    return True


def get_connection():
    """Return mock connection."""
    return type("Conn", (), {"close": lambda s: None, "commit": lambda s: None})()


def read_sql(query, conn=None):
    """Read data from session state."""
    q = query.lower()
    
    if "daily_log" in q:
        data = st.session_state.get("daily_log", [])
        if not data:
            return pd.DataFrame(columns=["id", "date", "compound_name", "dosage_mcg", "notes"])
        df = pd.DataFrame(data)
        if "ORDER BY date DESC" in query.upper():
            df = df.sort_values("date", ascending=False)
        if "LIMIT" in query.upper():
            try:
                limit = int(query.upper().split("LIMIT")[1].strip().split()[0])
                df = df.head(limit)
            except:
                pass
        return df[["id", "date", "compound_name", "dosage_mcg", "notes"]]
    
    elif "blood_markers" in q:
        data = st.session_state.get("blood_markers", [])
        cols = ["id", "date", "test_name", "igf1", "glucose", "free_testosterone",
                "total_testosterone", "estradiol", "cholesterol_total", "ldl", "hdl",
                "triglycerides", "alt", "ast", "tsh", "creatinine", "notes"]
        if not data:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(data)
        if "ORDER BY date ASC" in query.upper():
            df = df.sort_values("date", ascending=True)
        elif "ORDER BY date DESC" in query.upper():
            df = df.sort_values("date", ascending=False)
        return df[cols]
    
    elif "oracle_history" in q:
        data = st.session_state.get("oracle_history", [])
        if not data:
            return pd.DataFrame(columns=["id", "timestamp", "question", "answer", "source", "model_used"])
        df = pd.DataFrame(data)
        if "ORDER BY timestamp DESC" in query.upper():
            df = df.sort_values("timestamp", ascending=False)
        if "LIMIT" in query.upper():
            try:
                limit = int(query.upper().split("LIMIT")[1].strip().split()[0])
                df = df.head(limit)
            except:
                pass
        return df[["id", "timestamp", "question", "answer", "source", "model_used"]]
    
    return pd.DataFrame()


def execute_insert(table, params):
    """Insert data into session state."""
    idx = st.session_state.next_id.get(table, 1)
    
    if table == "daily_log":
        entry = {
            "id": idx,
            "date": params[0],
            "compound_name": params[1],
            "dosage_mcg": params[2],
            "notes": params[3]
        }
    elif table == "blood_markers":
        entry = {
            "id": idx,
            "date": params[0],
            "test_name": params[1],
            "igf1": params[2], "glucose": params[3],
            "free_testosterone": params[4], "total_testosterone": params[5],
            "estradiol": params[6], "cholesterol_total": params[7],
            "ldl": params[8], "hdl": params[9], "triglycerides": params[10],
            "alt": params[11], "ast": params[12], "tsh": params[13],
            "creatinine": params[14],
            "notes": params[15] if len(params) > 15 else None
        }
    elif table == "oracle_history":
        entry = {
            "id": idx,
            "timestamp": params[0],
            "question": params[1],
            "answer": params[2],
            "source": params[3],
            "model_used": params[4]
        }
    else:
        return False
    
    st.session_state[table].append(entry)
    st.session_state.next_id[table] = idx + 1
    return True


def execute_delete(table):
    """Delete all data from a table."""
    st.session_state[table] = []
    return True


# Initialize
init_database()

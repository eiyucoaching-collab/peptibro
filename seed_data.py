"""
Peptibro Cloud - Sample Data Generator
Run once to populate the database with sample data.
"""

import sqlite3
from datetime import date, timedelta
import os
from pathlib import Path

# Get database path (same logic as database_setup)
def _get_db_path():
    if os.getenv("STREAMLIT_CLOUD") or Path("/tmp").exists():
        return Path("/tmp") / "peptibro.db"
    return Path("db") / "peptibro.db"

DB_PATH = _get_db_path()


def seed_sample_data():
    """Insert sample data for testing."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Check if data already exists
    count = c.execute("SELECT COUNT(*) FROM blood_markers").fetchone()[0]
    if count > 0:
        print(f"Database already has {count} records. Skipping seed.")
        conn.close()
        return

    # 3 blood work analyses
    analyses = [
        ("2026-03-15", "Lab Central - Pre protocolo", 180, 88, 12.5, 520, 28, 190, 95, 55, 110, 25, 28, 2.1, 0.9),
        ("2026-06-01", "Lab Central - 3 meses BPC-157", 220, 82, 18.2, 610, 22, 185, 88, 60, 95, 22, 25, 1.8, 0.85),
        ("2026-06-29", "Lab Central - 4 meses protocolo", 245, 79, 21.0, 680, 20, 178, 82, 65, 88, 20, 23, 1.6, 0.82),
    ]

    for a in analyses:
        c.execute(
            """INSERT INTO blood_markers 
            (date, test_name, igf1, glucose, free_testosterone, total_testosterone,
             estradiol, cholesterol_total, ldl, hdl, triglycerides, alt, ast, tsh, creatinine)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            a,
        )

    # 30 days of daily log
    compounds = ["BPC-157", "TB-500", "CJC-1295", "Ipamorelin", "BPC-157"]
    doses = [250, 500, 2000, 200, 250]
    notes_list = [
        "Dolor hombro mejorando",
        "Recuperacion muscular OK",
        "Sueno profundo",
        "Energia estable",
        "Sin efectos adversos",
    ]

    for i in range(30):
        d = (date(2026, 6, 1) + timedelta(days=i)).isoformat()
        idx = i % len(compounds)
        c.execute(
            "INSERT INTO daily_log (date, compound_name, dosage_mcg, notes) VALUES (?, ?, ?, ?)",
            (d, compounds[idx], doses[idx], notes_list[idx]),
        )

    conn.commit()
    blood_count = c.execute("SELECT COUNT(*) FROM blood_markers").fetchone()[0]
    log_count = c.execute("SELECT COUNT(*) FROM daily_log").fetchone()[0]
    print(f"Seed complete: {blood_count} analyses, {log_count} daily logs")
    conn.close()


if __name__ == "__main__":
    seed_sample_data()

"""
Peptibro API Server
Expone funciones de Peptibro vía HTTP para OpenClaw
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openclaw_bridge import (
    get_recent_logs,
    get_recent_bloodwork,
    check_out_of_range,
    get_protocol_summary,
    REFERENCE_RANGES
)
import uvicorn
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# === LOGGING ===
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api_server.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("peptibro_api")

API_KEY = os.getenv("PEPTIBRO_API_KEY", "peptibro-local-2026")

app = FastAPI(title="Peptibro API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:*", "http://localhost:*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def verify_key(request: Request):
    key = request.headers.get("X-API-Key") or request.query_params.get("key")
    if key != API_KEY:
        logger.warning(f"Unauthorized access attempt from {request.client.host}")
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start).total_seconds()
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration:.3f}s)")
    return response


@app.get("/")
def root():
    return {"message": "Peptibro API", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/protocol/summary")
def protocol_summary(request: Request):
    verify_key(request)
    return {"summary": get_protocol_summary()}


@app.get("/protocol/logs")
def protocol_logs(request: Request, days: int = 30):
    verify_key(request)
    return {"logs": get_recent_logs(days)}


@app.get("/bloodwork/recent")
def bloodwork_recent(request: Request, limit: int = 5):
    verify_key(request)
    return {"bloodwork": get_recent_bloodwork(limit)}


@app.get("/bloodwork/alerts")
def bloodwork_alerts(request: Request):
    verify_key(request)
    bloodwork = get_recent_bloodwork(2)
    alerts = check_out_of_range(bloodwork)
    return {"alerts": alerts}


@app.get("/markers/ranges")
def marker_ranges(request: Request):
    verify_key(request)
    return {"ranges": REFERENCE_RANGES}


# === NEW ENDPOINTS FOR AUTOMATION ===

@app.get("/dose/check-today")
def check_today_doses(request: Request):
    """Check if today's doses have been logged."""
    verify_key(request)
    from database_setup import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute("SELECT COUNT(*) FROM daily_log WHERE date = ?", (today,))
    count = cursor.fetchone()[0]
    conn.close()
    logger.info(f"Dose check: {count} doses logged today")
    return {"date": today, "doses_logged": count, "has_dosed": count > 0}


@app.get("/dose/last")
def last_dose(request: Request):
    """Get the most recent dose entry."""
    verify_key(request)
    from database_setup import get_connection
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, compound_name, dosage_mcg, notes
        FROM daily_log ORDER BY id DESC LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"date": row[0], "compound": row[1], "dosage_mcg": row[2], "notes": row[3]}
    return {"date": None, "compound": None, "dosage_mcg": None, "notes": None}


@app.get("/summary/weekly")
def weekly_summary(request: Request):
    """Generate a weekly summary of doses and bloodwork."""
    verify_key(request)
    from database_setup import get_connection
    import pandas as pd

    conn = get_connection()
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()

    df_log = pd.read_sql("""
        SELECT date, compound_name, dosage_mcg, notes
        FROM daily_log WHERE date >= ? ORDER BY date DESC
    """, conn, params=(week_ago,))

    df_blood = pd.read_sql("""
        SELECT date, test_name, igf1, glucose, free_testosterone,
               total_testosterone, estradiol, alt, ast, tsh
        FROM blood_markers WHERE date >= ? ORDER BY date DESC
    """, conn, params=(week_ago,))
    conn.close()

    summary = {
        "period": f"{(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
        "total_doses": len(df_log),
        "compounds_used": df_log["compound_name"].unique().tolist() if not df_log.empty else [],
        "doses_by_compound": df_log.groupby("compound_name")["dosage_mcg"].sum().to_dict() if not df_log.empty else {},
        "bloodwork_count": len(df_blood),
        "has_alerts": False,
        "alerts": []
    }

    if not df_blood.empty:
        alerts = check_out_of_range(df_blood.to_dict("records"))
        summary["has_alerts"] = len(alerts) > 0
        summary["alerts"] = alerts

    logger.info(f"Weekly summary: {summary['total_doses']} doses, {summary['bloodwork_count']} bloodwork")
    return summary


@app.get("/rag/query")
def rag_query(request: Request, q: str = ""):
    """Query the RAG engine for peptide information."""
    verify_key(request)
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    from rag_engine import query_peptide_protocol
    logger.info(f"RAG query: {q}")
    result = query_peptide_protocol(q)
    return {"query": q, "result": result}


if __name__ == "__main__":
    logger.info("Starting Peptibro API Server")
    uvicorn.run(app, host="127.0.0.1", port=8000)

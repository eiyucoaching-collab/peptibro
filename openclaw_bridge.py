"""
Peptibro - OpenClaw Integration Bridge
Permite a OpenClaw interactuar con la base de datos y RAG de Peptibro
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path("C:/Users/FUJITSU/Peptibro/db/peptibro.db")

# Rangos de referencia
REFERENCE_RANGES = {
    "igf1": {"min": 100, "max": 300, "unit": "ng/mL"},
    "glucose": {"min": 70, "max": 99, "unit": "mg/dL"},
    "free_testosterone": {"min": 8.0, "max": 25.0, "unit": "pg/mL"},
    "total_testosterone": {"min": 300, "max": 1000, "unit": "ng/dL"},
    "estradiol": {"min": 10, "max": 40, "unit": "pg/mL"},
    "cholesterol_total": {"min": 0, "max": 200, "unit": "mg/dL"},
    "ldl": {"min": 0, "max": 100, "unit": "mg/dL"},
    "hdl": {"min": 40, "max": 999, "unit": "mg/dL"},
    "triglycerides": {"min": 0, "max": 150, "unit": "mg/dL"},
    "alt": {"min": 0, "max": 40, "unit": "U/L"},
    "ast": {"min": 0, "max": 40, "unit": "U/L"},
    "tsh": {"min": 0.4, "max": 4.0, "unit": "mIU/L"},
    "creatinine": {"min": 0.7, "max": 1.3, "unit": "mg/dL"},
}


def get_connection():
    return sqlite3.connect(str(DB_PATH))


def get_recent_logs(days=30):
    """Obtiene los últimos registros del log diario."""
    conn = get_connection()
    cursor = conn.cursor()
    
    start_date = (datetime.now() - timedelta(days=days)).isoformat()
    cursor.execute("""
        SELECT date, compound_name, dosage_mcg, notes
        FROM daily_log
        WHERE date >= ?
        ORDER BY date DESC, id DESC
    """, (start_date,))
    
    logs = cursor.fetchall()
    conn.close()
    
    return [
        {
            "date": log[0],
            "compound": log[1],
            "dosage_mcg": log[2],
            "notes": log[3]
        }
        for log in logs
    ]


def get_recent_bloodwork(limit=5):
    """Obtiene las últimas analíticas."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT date, test_name, igf1, glucose, free_testosterone, 
               total_testosterone, estradiol, cholesterol_total, ldl, 
               hdl, triglycerides, alt, ast, tsh, creatinine
        FROM blood_markers
        ORDER BY date DESC
        LIMIT ?
    """, (limit,))
    
    bloodwork = cursor.fetchall()
    conn.close()
    
    return [
        {
            "date": bw[0],
            "test_name": bw[1],
            "igf1": bw[2],
            "glucose": bw[3],
            "free_testosterone": bw[4],
            "total_testosterone": bw[5],
            "estradiol": bw[6],
            "cholesterol_total": bw[7],
            "ldl": bw[8],
            "hdl": bw[9],
            "triglycerides": bw[10],
            "alt": bw[11],
            "ast": bw[12],
            "tsh": bw[13],
            "creatinine": bw[14]
        }
        for bw in bloodwork
    ]


def check_out_of_range(bloodwork_data):
    """Verifica biomarcadores fuera de rango."""
    alerts = []
    
    for bw in bloodwork_data:
        for marker, ranges in REFERENCE_RANGES.items():
            value = bw.get(marker)
            if value is not None:
                try:
                    value = float(value)
                    if value < ranges["min"] or value > ranges["max"]:
                        alerts.append({
                            "date": bw["date"],
                            "marker": marker,
                            "value": value,
                            "min": ranges["min"],
                            "max": ranges["max"],
                            "unit": ranges["unit"]
                        })
                except (ValueError, TypeError):
                    pass
    
    return alerts


def get_protocol_summary():
    """Genera un resumen del protocolo activo."""
    logs = get_recent_logs(30)
    
    if not logs:
        return "No hay registros recientes en el log diario."
    
    # Agrupar por compuesto
    compounds = {}
    for log in logs:
        compound = log["compound"]
        if compound not in compounds:
            compounds[compound] = {
                "total_doses": 0,
                "total_mcg": 0,
                "last_date": log["date"],
                "notes": []
            }
        compounds[compound]["total_doses"] += 1
        compounds[compound]["total_mcg"] += log["dosage_mcg"]
        if log["notes"]:
            compounds[compound]["notes"].append(log["notes"])
    
    # Generar resumen
    summary = "📋 RESUMEN DE PROTOCOLO (últimos 30 días)\n\n"
    
    for compound, data in compounds.items():
        avg_dose = data["total_mcg"] / data["total_doses"]
        summary += f"• {compound}: {data['total_doses']} dosis, promedio {avg_dose:.0f} mcg\n"
    
    # Alertas de analíticas
    bloodwork = get_recent_bloodwork(2)
    alerts = check_out_of_range(bloodwork)
    
    if alerts:
        summary += "\n🚨 ALERTAS DE ANALÍTICAS:\n"
        for alert in alerts:
            summary += f"  • {alert['marker'].upper()}: {alert['value']} {alert['unit']} "
            summary += f"(rango: {alert['min']}-{alert['max']}) - {alert['date']}\n"
    
    return summary


if __name__ == "__main__":
    print("=== Peptibro OpenClaw Bridge ===")
    print("\n1. Protocolo reciente:")
    print(get_protocol_summary())
    
    print("\n2. Últimas analíticas:")
    bloodwork = get_recent_bloodwork(3)
    for bw in bloodwork:
        print(f"  {bw['date']} ({bw['test_name']})")
    
    print("\n3. Alertas fuera de rango:")
    alerts = check_out_of_range(bloodwork)
    for alert in alerts:
        print(f"  🚨 {alert['marker']}: {alert['value']} {alert['unit']}")

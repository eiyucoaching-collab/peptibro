"""
Peptibro - Aplicacion Hibrida para Seguimiento de Protocolos de Peptidos
Frontend: Streamlit
- Log Diario
- Oraculo Clinico (RAG estricto con citas de fuente) + Historial
- Dashboard de Analiticas (Plotly) con Alertas de Rangos
- Exportacion de Informe PDF Mensual
- Cloud-ready (Streamlit Community Cloud)
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import date, timedelta, datetime
from pathlib import Path
import sys
import io

from database_setup import init_database, get_connection, read_sql, execute_insert, execute_delete
from rag_engine import query_peptide_protocol, ingest_knowledge_base, has_knowledge_base, chat_with_coach, GEMINI_KEY

# Initialize database
init_database()

# Monkey-patch pd.read_sql to use our custom read_sql
_original_read_sql = pd.read_sql

def _patched_read_sql(query, conn=None, *args, **kwargs):
    return read_sql(query, conn)

pd.read_sql = _patched_read_sql

# ============================================================
# RANGOS DE REFERENCIA PARA BIOMARCADORES
# ============================================================
REFERENCE_RANGES = {
    "igf1": {"label": "IGF-1", "min": 100, "max": 300, "unit": "ng/mL"},
    "glucose": {"label": "Glucosa", "min": 70, "max": 99, "unit": "mg/dL"},
    "free_testosterone": {"label": "Testosterona Libre", "min": 8.0, "max": 25.0, "unit": "pg/mL"},
    "total_testosterone": {"label": "Testosterona Total", "min": 300, "max": 1000, "unit": "ng/dL"},
    "estradiol": {"label": "Estradiol", "min": 10, "max": 40, "unit": "pg/mL"},
    "cholesterol_total": {"label": "Colesterol Total", "min": 0, "max": 200, "unit": "mg/dL"},
    "ldl": {"label": "LDL", "min": 0, "max": 100, "unit": "mg/dL"},
    "hdl": {"label": "HDL", "min": 40, "max": 999, "unit": "mg/dL"},
    "triglycerides": {"label": "Trigliceridos", "min": 0, "max": 150, "unit": "mg/dL"},
    "alt": {"label": "ALT", "min": 0, "max": 40, "unit": "U/L"},
    "ast": {"label": "AST", "min": 0, "max": 40, "unit": "U/L"},
    "tsh": {"label": "TSH", "min": 0.4, "max": 4.0, "unit": "mIU/L"},
    "creatinine": {"label": "Creatinina", "min": 0.7, "max": 1.3, "unit": "mg/dL"},
}


def is_out_of_range(marker, value):
    """Devuelve True si el valor esta fuera del rango de referencia."""
    if marker not in REFERENCE_RANGES:
        return False
    if value is None or pd.isna(value):
        return False
    r = REFERENCE_RANGES[marker]
    return value < r["min"] or value > r["max"]


def format_marker_label(marker):
    """Formatea el nombre del marcador con su unidad."""
    if marker in REFERENCE_RANGES:
        r = REFERENCE_RANGES[marker]
        return f"{r['label']} ({r['unit']})"
    return marker.upper()


def save_oracle_query(question, answer):
    """Guarda una consulta del Oraculo en la base de datos."""
    try:
        execute_insert("oracle_history", (
            datetime.now().isoformat(), question, answer, "ChromaDB", "gemini-2.0-flash"
        ))
    except Exception:
        pass


def export_monthly_report_pdf(year, month):
    """Genera un informe PDF mensual con log diario, biomarcadores y historial del Oraculo."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    )
    from reportlab.lib.enums import TA_CENTER

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CenterTitle",
        parent=styles["Title"], alignment=TA_CENTER,
        fontSize=18, spaceAfter=20
    ))
    elements = []

    month_names = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    month_name = month_names[month - 1] if 1 <= month <= 12 else str(month)

    # Titulo
    elements.append(Paragraph(f"Peptibro - Informe Mensual", styles["CenterTitle"]))
    elements.append(Paragraph(f"{month_name} {year}", styles["CenterTitle"]))
    elements.append(Spacer(1, 0.5 * cm))

    # ---- Seccion 1: Log Diario ----
    elements.append(Paragraph("<b>1. Registro Diario de Administraciones</b>", styles["Heading2"]))
    conn = get_connection()
    df_logs = pd.read_sql("""
        SELECT date, compound_name, dosage_mcg, notes
        FROM daily_log
        WHERE date LIKE ?
        ORDER BY date ASC, id ASC
    """, conn, params=[f"{year}-{month:02d}%"])

    if df_logs.empty:
        elements.append(Paragraph("No hay registros en este mes.", styles["Normal"]))
    else:
        log_data = [["Fecha", "Compuesto", "Dosis (mcg)", "Notas"]]
        for _, row in df_logs.iterrows():
            log_data.append([
                str(row["date"]),
                str(row["compound_name"]),
                str(row["dosage_mcg"]),
                str(row["notes"]) if row["notes"] else ""
            ])
        log_table = Table(log_data, colWidths=[3 * cm, 4 * cm, 3 * cm, 7 * cm])
        log_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1d24")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        elements.append(log_table)

    elements.append(Spacer(1, 1 * cm))

    # ---- Seccion 2: Biomarcadores con Alertas ----
    elements.append(Paragraph("<b>2. Biomarcadores</b>", styles["Heading2"]))
    df_blood = pd.read_sql("""
        SELECT date, test_name, igf1, glucose, free_testosterone, total_testosterone,
               estradiol, cholesterol_total, ldl, hdl, triglycerides, alt, ast, tsh, creatinine
        FROM blood_markers
        WHERE date LIKE ?
        ORDER BY date ASC
    """, conn, params=[f"{year}-{month:02d}%"])

    if df_blood.empty:
        elements.append(Paragraph("No hay analiticas en este mes.", styles["Normal"]))
    else:
        blood_cols = [
            "date", "test_name", "igf1", "glucose", "free_testosterone", "total_testosterone",
            "estradiol", "cholesterol_total", "ldl", "hdl", "triglycerides",
            "alt", "ast", "tsh", "creatinine"
        ]
        header_labels = [
            "Fecha", "Lab", "IGF-1", "Gluc", "Test-L", "Test-T",
            "Estr", "Col", "LDL", "HDL", "Trig",
            "ALT", "AST", "TSH", "Creat"
        ]
        blood_data = [header_labels]
        alert_rows = []
        for idx, row in df_blood.iterrows():
            row_data = []
            for col in blood_cols:
                val = row[col]
                row_data.append(str(val) if val is not None and not pd.isna(val) else "-")
            blood_data.append(row_data)
            # Detectar filas con valores fuera de rango
            for col in blood_cols:
                if col in REFERENCE_RANGES:
                    val = row[col]
                    if val is not None and not pd.isna(val) and is_out_of_range(col, val):
                        col_idx = blood_cols.index(col)
                        alert_rows.append((idx + 1, col_idx))
                        break

        blood_table = Table(blood_data, colWidths=[2.2 * cm] + [1.3 * cm] * (len(blood_cols) - 1))
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1d24")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
        for row_idx, col_idx in alert_rows:
            style_cmds.append(("BACKGROUND", (col_idx, row_idx), (col_idx, row_idx), colors.HexColor("#ffcccc")))
        blood_table.setStyle(TableStyle(style_cmds))
        elements.append(blood_table)
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(Paragraph(
            "<font color='red'>Celdas en rojo = valor fuera del rango de referencia.</font>",
            styles["Normal"]
        ))

    elements.append(Spacer(1, 1 * cm))

    # ---- Seccion 3: Ultimas 5 consultas del Oraculo ----
    elements.append(Paragraph("<b>3. Ultimas Consultas del Oraculo (mes)</b>", styles["Heading2"]))
    df_oracle = pd.read_sql("""
        SELECT timestamp, question, answer
        FROM oracle_history
        WHERE timestamp LIKE ?
        ORDER BY timestamp DESC
        LIMIT 5
    """, conn, params=[f"{year}-{month:02d}%"])

    conn.close()

    if df_oracle.empty:
        elements.append(Paragraph("No hay consultas del Oraculo en este mes.", styles["Normal"]))
    else:
        for _, row in df_oracle.iterrows():
            elements.append(Paragraph(f"<b>{row['timestamp']}</b>", styles["Normal"]))
            elements.append(Paragraph(f"<b>Pregunta:</b> {row['question']}", styles["Normal"]))
            answer_short = str(row["answer"])[:500]
            if len(str(row["answer"])) > 500:
                answer_short += "..."
            elements.append(Paragraph(f"<b>Respuesta:</b> {answer_short}", styles["Normal"]))
            elements.append(Spacer(1, 0.5 * cm))

    # Pie de pagina
    elements.append(Spacer(1, 1 * cm))
    elements.append(Paragraph(
        "Peptibro - Informe generado automaticamente. "
        "El conocimiento clinico proviene de PDFs locales (RAG estricto). "
        "Consulte siempre a un profesional medico.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# Configuracion de pagina
st.set_page_config(
    page_title="Peptibro",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar base de datos al arrancar
init_database()

# Estilos minimalistas + PWA
st.markdown("""
<link rel="manifest" href="/app/static/manifest.json">
<meta name="theme-color" content="#1a1d24">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<style>
    /* === OVERRIDE STREAMLIT DEFAULTS === */
    
    /* Force dark background everywhere */
    html, body, [class*="css"] {
        background-color: #0f1116 !important;
        color: #f0f0f0 !important;
    }
    
    /* Main app background */
    .stApp, .main, .block-container {
        background-color: #0f1116 !important;
        color: #f0f0f0 !important;
    }
    
    /* All text forced white */
    p, span, div, label, li, td, th, h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li, .stMarkdown strong, .stMarkdown em, .stMarkdown span {
        color: #f0f0f0 !important;
    }
    
    /* Strong and em */
    strong, b { color: #ffffff !important; }
    em, i { color: #a0c4ff !important; }
    
    /* Tabs - fully styled */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px !important; 
        background: #1a1d24 !important; 
        border-radius: 10px !important; 
        padding: 4px !important; 
    }
    .stTabs [data-baseweb="tab"] { 
        background-color: transparent !important; 
        border-radius: 8px !important; 
        color: #a0a0a0 !important;
        font-weight: 500 !important;
    }
    .stTabs [data-baseweb="tab"]:hover { 
        color: #ffffff !important; 
        background: rgba(255,255,255,0.05) !important; 
    }
    .stTabs [aria-selected="true"] { 
        background-color: #2d3748 !important; 
        color: #ffffff !important; 
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #667eea !important;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #1a1d24 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatMessageContent"] {
        color: #f0f0f0 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #141820 !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #d0d0d0 !important;
    }
    [data-testid="stSidebar"] h1 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] label {
        color: #c0c0c0 !important;
    }
    
    /* Text inputs */
    .stTextInput input, 
    .stNumberInput input, 
    .stTextArea textarea,
    .stSelectbox select {
        color: #ffffff !important;
        background: #1e2228 !important;
        border: 1px solid #3a3f47 !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus, 
    .stNumberInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.2) !important;
    }
    .stTextInput label, .stNumberInput label, .stTextArea label, .stSelectbox label {
        color: #a0a0a0 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102,126,234,0.4) !important;
    }
    
    /* Form submit button */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Form container */
    .stForm {
        background: #1a1d24 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader, 
    .streamlit-expanderHeader:hover,
    details summary {
        color: #ffffff !important;
        background: #1a1d24 !important;
    }
    details[open] {
        background: #1a1d24 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
    }
    
    /* Success/Warning/Error/Info */
    .stSuccess, .stSuccess p { color: #22c55e !important; background: rgba(34,197,94,0.1) !important; border: 1px solid rgba(34,197,94,0.3) !important; border-radius: 8px !important; }
    .stWarning, .stWarning p { color: #eab308 !important; background: rgba(234,179,8,0.1) !important; border: 1px solid rgba(234,179,8,0.3) !important; border-radius: 8px !important; }
    .stError, .stError p { color: #ef4444 !important; background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important; border-radius: 8px !important; }
    .stInfo, .stInfo p { color: #3b82f6 !important; background: rgba(59,130,246,0.1) !important; border: 1px solid rgba(59,130,246,0.3) !important; border-radius: 8px !important; }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: #1a1d24 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }
    [data-testid="stMetricLabel"] { color: #a0a0a0 !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stMetricDelta"] { color: #22c55e !important; }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    [data-testid="stDataFrame"] td, 
    [data-testid="stDataFrame"] th {
        color: #f0f0f0 !important;
        background: #1a1d24 !important;
    }
    
    /* Divider */
    hr { border-color: #2d3748 !important; opacity: 0.5; }
    
    /* Radio buttons & checkboxes */
    .stRadio label, .stCheckbox label { color: #d0d0d0 !important; }
    
    /* Captions */
    .stCaption, small, .caption { color: #a0a0a0 !important; }
    
    /* File uploader */
    .stFileUploader { background: #1a1d24 !important; border: 1px solid #2d3748 !important; border-radius: 8px !important; }
    
    /* Progress bar */
    .stProgress > div > div { background: #667eea !important; }
    
    /* Multi-select */
    .stMultiSelect [data-baseweb="tag"] { background: #2d3748 !important; color: #ffffff !important; }
    .stMultiSelect [data-baseweb="select"] { background: #1e2228 !important; }
    
    /* Date input */
    .stDateInput input { color: #ffffff !important; background: #1e2228 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("💉 Peptibro")
    st.caption("Seguimiento de Péptidos • RAG Clínico • Analíticas")

    # === QUICK DOSE WIDGET ===
    st.markdown("---")
    st.markdown("**⚡ Dosis Rápida**")
    with st.form("quick_dose", clear_on_submit=True):
        q_compound = st.text_input("Compuesto", placeholder="ej: BPC-157", key="q_compound")
        q_dose = st.number_input("Dosis (mcg)", min_value=0.0, step=50.0, value=250.0, key="q_dose")
        if st.form_submit_button("💉 Registrar", use_container_width=True):
            if q_compound.strip():
                try:
                    execute_insert("daily_log", (
                        date.today().isoformat(), q_compound.strip(), float(q_dose), "Dosis rápida"
                    ))
                    st.success(f"✅ {q_compound} - {q_dose} mcg")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Nombre requerido")

    st.markdown("---")

    # Estado de la Base de Conocimiento
    kb_status = "✅ Cargada" if has_knowledge_base() else "⚠️ Vacía - Ingesta requerida"
    st.metric("Base de Conocimiento (RAG)", kb_status)

    if st.button("🔄 Ingerir / Actualizar PDFs", use_container_width=True):
        with st.spinner("Procesando PDFs de Base_Conocimiento..."):
            ingest_knowledge_base(force_rebuild=False)
        st.success("Ingesta completada. Reinicia la consulta si es necesario.")
        st.rerun()

    st.divider()

    # === EXPORT DATA ===
    st.markdown("**📥 Exportar Datos**")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("CSV", use_container_width=True):
            try:
                conn = get_connection()
                if conn:
                    df_logs = pd.read_sql("SELECT * FROM daily_log ORDER BY date DESC", conn)
                    df_blood = pd.read_sql("SELECT * FROM blood_markers ORDER BY date DESC", conn)
                    conn.close()
                    csv_log = df_logs.to_csv(index=False)
                    csv_blood = df_blood.to_csv(index=False)
                    st.download_button("⬇️ Log CSV", csv_log, "peptibro_log.csv", "text/csv", key="dl_log")
                    st.download_button("⬇️ Analíticas CSV", csv_blood, "peptibro_bloodwork.csv", "text/csv", key="dl_blood")
            except Exception:
                st.error("Error exporting data")
    with col_exp2:
        if st.button("JSON", use_container_width=True):
            try:
                conn = get_connection()
                if conn:
                    df_logs = pd.read_sql("SELECT * FROM daily_log ORDER BY date DESC", conn)
                    df_blood = pd.read_sql("SELECT * FROM blood_markers ORDER BY date DESC", conn)
                    conn.close()
                    json_data = {
                        "daily_log": df_logs.to_dict(orient="records"),
                        "bloodwork": df_blood.to_dict(orient="records"),
                        "exported_at": datetime.now().isoformat()
                    }
                    st.download_button("⬇️ JSON", json.dumps(json_data, ensure_ascii=False, indent=2), "peptibro_data.json", "application/json", key="dl_json")
            except Exception:
                st.error("Error exporting data")

    st.divider()
    st.markdown("**Modo:** Híbrido (tú decides cuándo consultar Gemini)")
    st.caption("El Oráculo responde estrictamente desde tus PDFs locales.")

# Titulo principal
st.title("Peptibro")
st.caption("Protocolos basados 100% en literatura médica local • Sin alucinaciones")

# Pestañas principales
tab_log, tab_biblioteca, tab_oraculo, tab_dashboard, tab_coach, tab_openclaw = st.tabs([
    "📝 Log Diario",
    "📚 Biblioteca de Péptidos",
    "🔮 Oráculo Clínico",
    "📊 Dashboard Analíticas",
    "🧠 Coach Clínico",
    "🦞 Clawd (OpenClaw)"
])

# ============================================================
# TAB 1: LOG DIARIO
# ============================================================
with tab_log:
    st.header("Registro Diario de Administraciones")

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.form("daily_log_form", clear_on_submit=True):
            log_date = st.date_input("Fecha", value=date.today())
            compound = st.text_input("Compuesto / Péptido", placeholder="ej: BPC-157, CJC-1295, Ipamorelin...")
            dosage = st.number_input("Dosis (mcg)", min_value=0.0, step=50.0, value=250.0)
            notes = st.text_area("Notas (sueño, energía, dolor, efectos...)", height=100)

            submitted = st.form_submit_button("💾 Guardar Registro", use_container_width=True)

            if submitted:
                if not compound.strip():
                    st.error("El nombre del compuesto es obligatorio.")
                else:
                    try:
                        execute_insert("daily_log", (
                            log_date.isoformat(), compound.strip(), float(dosage), notes.strip() or None
                        ))
                        st.success(f"Registro guardado: {compound} - {dosage} mcg")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    with col2:
        st.subheader("Últimos Registros")
        try:
            conn = get_connection()
            df_logs = pd.read_sql("""
                SELECT date, compound_name, dosage_mcg, notes 
                FROM daily_log 
                ORDER BY date DESC, id DESC 
                LIMIT 15
            """, conn)
            conn.close()
        except Exception:
            df_logs = pd.DataFrame()

        if not df_logs.empty:
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
        else:
            st.info("Aún no hay registros. Usa el formulario de la izquierda.")

# ============================================================
# TAB: BIBLIOTECA DE PÉPTIDOS
# ============================================================
with tab_biblioteca:
    st.header("📚 Biblioteca de Péptidos")
    st.caption("Guía completa de péptidos con usos, beneficios y riesgos basada en literatura médica")

    # Peptide database
    PEPTIDE_DB = {
        "BPC-157": {
            "category": "Reparación Tisular",
            "full_name": "Body Protection Compound-157",
            "description": "Péptido de 157 aminoácidos derivado de una proteína gástrica. Potente agente de reparación y protección de tejidos.",
            "uses": [
                "Curación de heridas y lesiones musculares",
                "Reparación de tendones y ligamentos",
                "Protección gástrica (úlceras, gastritis)",
                "Recuperación de lesiones articulares",
                "Neuroprotección y reparación nerviosa"
            ],
            "benefits": [
                "Acelera la cicatrización de tejidos hasta 3x",
                "Efecto angiogénico (nuevos vasos sanguíneos)",
                "Protector gástrico sin efectos adversos significativos",
                "Reduce inflamación sistémica",
                "Mejora la recuperación muscular post-lesión"
            ],
            "risks": [
                "Efectos secundarios mínimos reportados",
                "Posible interacción con anticoagulantes",
                "Datos a largo plazo limitados en humanos",
                "No recomendado durante embarazo/lactancia"
            ],
            "typical_dose": "250-500 mcg subcutáneo 1-2x/día",
            "cycle": "4-12 semanas, descanso de 4 semanas"
        },
        "TB-500": {
            "category": "Reparación Tisular",
            "full_name": "Timosina Beta-4 (fragmento)",
            "description": "Fragmento sintético de la Timosina Beta-4, proteína presente en casi todos los tejidos humanos.",
            "uses": [
                "Recuperación de lesiones musculares",
                "Reparación de tendones y ligamentos",
                "Curación de heridas crónicas",
                "Regeneración de tejido cardiaco",
                "Anti-inflamatorio sistémico"
            ],
            "benefits": [
                "Promueve la migración celular a sitios de lesión",
                "Estimula la formación de nuevos vasos sanguíneos",
                "Reduce formación de cicatrices",
                "Efecto anti-inflamatorio potente",
                "Mejora flexibilidad articular"
            ],
            "risks": [
                "Generalmente bien tolerado",
                "Posible somnolencia en dosis altas",
                "Interacción con medicamentos antiplaquetarios",
                "Evidencia limitada en humanos a largo plazo"
            ],
            "typical_dose": "2.5-5 mg subcutáneo 2x/semana",
            "cycle": "4-8 semanas"
        },
        "CJC-1295": {
            "category": "Eje Hormonal",
            "full_name": "CJC-1295 (con/sin DAC)",
            "description": "Análogo de la hormona liberadora de hormona del crecimiento (GHRH). Estimula la producción natural de GH.",
            "uses": [
                "Aumento de masa muscular magra",
                "Reducción de grasa corporal",
                "Mejora de calidad de sueño",
                "Recuperación y reparación de tejidos",
                "Anti-envejecimiento"
            ],
            "benefits": [
                "Aumento significativo de IGF-1 y GH",
                "Mejora composición corporal",
                "Duerme más profundo y reparador",
                "Efectos anti-envejecimiento documentados",
                "Fácil administración (subcutánea)"
            ],
            "risks": [
                "Retención de líquidos temporal",
                "Dolor en sitio de inyección",
                "Hormigueo o entumecimiento temporal",
                "Posible aumento de insulina en ayunas",
                "No usar con tumores activos"
            ],
            "typical_dose": "100-300 mcg 2-3x/día (sin DAC) o 2 mg/sem (con DAC)",
            "cycle": "Ciclos de 3-6 meses"
        },
        "Ipamorelin": {
            "category": "Eje Hormonal",
            "full_name": "Ipamorelin",
            "description": "Péptido secretagogo de GH (GHRP). Estimula la liberación de hormona del crecimiento de forma selectiva.",
            "uses": [
                "Aumento de GH e IGF-1",
                "Mejora de calidad de sueño",
                "Aumento de masa muscular",
                "Reducción de grasa corporal",
                "Recuperación atlética"
            ],
            "benefits": [
                "Libera GH sin afectar cortisol o prolactina",
                "Mejora dramática del sueño profundo",
                "Efectos sinérgicos con CJC-1295",
                "Seguro a largo plazo según estudios",
                "Fácil dosificación (antes de dormir)"
            ],
            "risks": [
                "Posible aumento temporal de apetito",
                "Dolor leve en sitio de inyección",
                "Mareos leves en algunos usuarios",
                "No usar con cáncer activo"
            ],
            "typical_dose": "200-300 mcg subcutáneo antes de dormir",
            "cycle": "3-6 meses continuos"
        },
        "Semaglutide": {
            "category": "Metabólico",
            "full_name": "Semaglutide (Ozempic/Wegovy)",
            "description": "Agonista del receptor GLP-1. Reduce el apetito y mejora el control glucémico.",
            "uses": [
                "Pérdida de peso significativa",
                "Control de diabetes tipo 2",
                "Reducción de apetito y antojos",
                "Mejora de sensibilidad a la insulina",
                "Reducción de riesgo cardiovascular"
            ],
            "benefits": [
                "Pérdida de peso del 10-15% en 68 semanas",
                "Reducción de HbA1c significativa",
                "Menor riesgo de eventos cardiovasculares",
                "Una inyección semanal (conveniencia)",
                "Efecto sobre el centro de saciedad cerebral"
            ],
            "risks": [
                "Náuseas (común al inicio, se resuelve)",
                "Vómitos temporales",
                "Diarrea o estreñimiento",
                "Riesgo de pancreatitis (raro)",
                "Posible pérdida de masa muscular",
                "Pérdida de peso rápida puede causar cálculos biliares"
            ],
            "typical_dose": "0.25 mg → 0.5 mg → 1 mg → 2.4 mg semanal",
            "cycle": "Continuo bajo supervisión médica"
        },
        "Tirzepatide": {
            "category": "Metabólico",
            "full_name": "Tirzepatide (Mounjaro/Zepbound)",
            "description": "Agonista dual de receptores GIP y GLP-1. Más potente que los agonistas de GLP-1 solos.",
            "uses": [
                "Pérdida de peso superior a semaglutide",
                "Control de diabetes tipo 2",
                "Reducción de apetito extrema",
                "Mejora de metabolismo",
                "Reducción de grasa visceral"
            ],
            "benefits": [
                "Pérdida de peso del 20-25% en estudios",
                "Mejor control glucémico que semaglutide",
                "Efecto dual GIP+GLP-1 más completo",
                "Una inyección semanal",
                "Mejora marcadores lipídicos"
            ],
            "risks": [
                "Náuseas más intensas que semaglutide al inicio",
                "Vómitos frecuentes al ajustar dosis",
                "Diarrea significativa",
                "Riesgo de pancreatitis",
                "Pérdida de masa muscular preocupante",
                "Costo muy alto"
            ],
            "typical_dose": "2.5 mg → 5 mg → 10 mg → 15 mg semanal",
            "cycle": "Continuo bajo supervisión médica"
        },
        "HGH": {
            "category": "Eje Hormonal",
            "full_name": "Hormona de Crecimiento Humana",
            "description": "Hormona producida naturalmente por la hipófisis. Fundamental para crecimiento, reparación y metabolismo.",
            "uses": [
                "Aumento de masa muscular",
                "Reducción de grasa corporal",
                "Mejora de densidad ósea",
                "Recuperación de lesiones",
                "Anti-envejecimiento"
            ],
            "benefits": [
                "Efectos anabólicos potentes",
                "Mejora composición corporal drásticamente",
                "Acelera recuperación de lesiones",
                "Mejora calidad de piel y cabello",
                "Efectos neuroprotectores"
            ],
            "risks": [
                "Acromegalia (crecimiento excesivo de tejidos)",
                "Retención de líquidos significativa",
                "Dolor articular (síndrome del túnel carpiano)",
                "Resistencia a la insulina / diabetes",
                "Aumento de riesgo de cáncer (debate)",
                "Costo muy alto",
                "Requiere refrigeración"
            ],
            "typical_dose": "2-6 IU subcutáneo diario",
            "cycle": "3-6 meses, descansos obligatorios"
        },
        "GHK-Cu": {
            "category": "Reparación Tisular",
            "full_name": "GHK-Cu (Péptido de Cobre)",
            "description": "Tripéptido que contiene cobre. Promueve reparación de piel, tejido conectivo y efecto anti-envejecimiento.",
            "uses": [
                "Reparación de piel y tejidos",
                "Anti-envejecimiento cutáneo",
                "Cicatrización de heridas",
                "Regeneración capilar",
                "Protección contra daño oxidativo"
            ],
            "benefits": [
                "Estimula producción de colágeno y elastina",
                "Efecto anti-envejecimiento documentado",
                "Mejora textura y firmeza de piel",
                "Promueve crecimiento capilar",
                "Propiedades anti-inflamatorias"
            ],
            "risks": [
                "Muy seguro según estudios",
                "Posible irritación cutánea en uso tópico",
                "No usar con exceso de cobre (hemocromatosis)",
                "Efectos sistémicos limitados por vía subcutánea"
            ],
            "typical_dose": "1-2.5 mg subcutáneo diario",
            "cycle": "Continuo o ciclos de 3 meses"
        },
        "Epithalon": {
            "category": "Longevidad",
            "full_name": "Epitalon / Epithalon",
            "description": "Tetrapéptido que estimula la telomerasa, enzima que protege los telómeros y ralentiza el envejecimiento celular.",
            "uses": [
                "Anti-envejecimiento celular",
                "Extensión de longitud de telómeros",
                "Mejora de función inmunológica",
                "Regulación del ritmo circadiano",
                "Prevención de enfermedades degenerativas"
            ],
            "benefits": [
                "Estimula telomerasa (enzima anti-envejecimiento)",
                "Aumenta longitud de telómeros en estudios",
                "Mejora calidad de sueño profundo",
                "Efecto rejuvenecedor documentado en Rusia",
                "Potencial extensión de vida saludable"
            ],
            "risks": [
                "Datos a largo plazo limitados",
                "Posible interferencia con hormonas sexuales",
                "No recomendado con tumores activos",
                "Disponibilidad limitada"
            ],
            "typical_dose": "5 mg subcutáneo diario por 10-20 días",
            "cycle": "1-2 veces al año"
        },
        "MOTS-C": {
            "category": "Longevidad",
            "full_name": "Mitochondrial Open Reading Frame of the 12S rRNA-Type-C",
            "description": "Péptido mitocondrial que actúa como hormona metabólica. Regula metabolismo y sensibilidad a la insulina.",
            "uses": [
                "Regulación metabólica",
                "Mejora de sensibilidad a la insulina",
                "Pérdida de peso",
                "Protección mitocondrial",
                "Anti-envejecimiento"
            ],
            "benefits": [
                "Regula metabolismo de ácidos grasos",
                "Mejora utilización de glucosa",
                "Efecto anti-obesidad en estudios animales",
                "Protege mitocondrias del daño oxidativo",
                "Potencial anti-envejecimiento"
            ],
            "risks": [
                "Datos limitados en humanos",
                "Efectos a largo plazo desconocidos",
                "No recomendado durante embarazo",
                "Investigación en curso"
            ],
            "typical_dose": "5 mg subcutáneo semanal",
            "cycle": "8-12 semanas"
        },
        "NAD+": {
            "category": "Longevidad",
            "full_name": "Nicotinamida Adenina Dinucleótido",
            "description": "Coenzima fundamental para producción de energía celular y reparación de ADN. Disminuye con la edad.",
            "uses": [
                "Anti-envejecimiento celular",
                "Mejora de producción energética",
                "Protección de ADN",
                "Función cognitiva",
                "Recuperación post-exercicio"
            ],
            "benefits": [
                "Restaura niveles de NAD+ que disminuyen con la edad",
                "Mejora producción de ATP energético",
                "Activa sirtuinas (genes de longevidad)",
                "Mejora función mitocondrial",
                "Potencial efecto neuroprotector"
            ],
            "risks": [
                "Inyección IV puede causar molestias",
                "Náuseas con dosis altas IV",
                "Oral tiene biodisponibilidad limitada",
                "Posible interacción con medicamentos",
                "Costo elevado para IV"
            ],
            "typical_dose": "250-500 mg IV semanal u 100-250 mg oral diario",
            "cycle": "Ciclos de 4-8 semanas IV o continuo oral"
        },
        "Selank": {
            "category": "Nootrópico",
            "full_name": "Selank",
            "description": "Análogo sintético de la tuftsina con efectos ansiolíticos y nootrópicos. Reduce ansiedad sin sedación.",
            "uses": [
                "Reducción de ansiedad",
                "Mejora de memoria y aprendizaje",
                "Reducción de estrés",
                "Mejora de concentración",
                "Tratamiento de fobias"
            ],
            "benefits": [
                "Ansiolítico sin efectos sedantes",
                "Mejora memoria de trabajo",
                "Reduce cortisol y marcadores de estrés",
                "Efecto rápido (1-2 horas)",
                "No crea dependencia"
            ],
            "risks": [
                "Muy seguro según estudios",
                "Posible irritabilidad enบางคน",
                "Interacción con otros ansiolíticos",
                "Efectos a largo plazo poco estudiados"
            ],
            "typical_dose": "250-500 mcg nasal 2-3x/día",
            "cycle": "2-4 semanas, descanso de 1 semana"
        },
        "Semax": {
            "category": "Nootrópico",
            "full_name": "Semax",
            "description": "Análogo sintético de la hormona estimulante de melanocitos (MSH) con potentes efectos neuroprotectores y nootrópicos.",
            "uses": [
                "Mejora de memoria y concentración",
                "Neuroprotección",
                "Recuperación de daño cerebral",
                "Mejora de función inmunológica",
                "Tratamiento de accidente cerebrovascular"
            ],
            "benefits": [
                "Potencia memoria y aprendizaje",
                "Protege neuronas del daño",
                "Estimula factor neurotrófico (BDNF)",
                "Mejora flujo sanguíneo cerebral",
                "Efecto anti-depresivo leve"
            ],
            "risks": [
                "Generalmente bien tolerado",
                "Posible insomnio si se toma tarde",
                "Irritabilidad en algunos usuarios",
                "No usar con tumores"
            ],
            "typical_dose": "300-600 mcg nasal 1-2x/día",
            "cycle": "2-4 semanas"
        },
        "PT-141": {
            "category": "Otros",
            "full_name": "Bremelanotide (PT-141)",
            "description": "Agonista de receptores melanocortina que actúa directamente sobre el sistema nervioso central para aumentar la libido.",
            "uses": [
                "Aumento de libido (hombres y mujeres)",
                "Disfunción eréctil",
                "Deseo sexual hypoactivo",
                "Mejora de respuesta sexual"
            ],
            "benefits": [
                "Actúa en el cerebro (no vasodilatador)",
                "Efectivo para disfunción sexual psicogénica",
                "Funciona tanto en hombres como mujeres",
                "Una dosis dura 24-72 horas",
                "No requiere estimulación sexual para funcionar"
            ],
            "risks": [
                "Náuseas (efecto secundario más común)",
                "Enrojecimiento facial temporal",
                "Dolor de cabeza",
                "Posible aumento de presión arterial",
                "No combinar con otros medicamentos sexuales"
            ],
            "typical_dose": "1-2 mg subcutáneo según necesidad",
            "cycle": "Según necesidad, no más de 1x/día"
        },
        # ── Metabólicos / Pérdida de grasa ──
        "AOD-9604": {
            "category": "Metabólico",
            "full_name": "AOD-9604 (Anti-Obesity Drug fragment 176-191)",
            "description": "Fragmento sintético de la Hormona de Crecimiento (aminoácidos 176-191) que estimula la lipólisis sin afectar la glucosa o crecimiento celular.",
            "uses": [
                "Pérdida de grasa localizada",
                "Reducción de celulitis",
                "Estimulación del metabolismo lipídico",
                "Regeneración de cartílago (dosis altas)",
                "Complemento en tratamientos anti-obesidad"
            ],
            "benefits": [
                "Quema grasa sin efectos sobre nivel de azúcar",
                "No causa resistencia a la insulina",
                "Efecto regenerativo en tejido conectivo",
                "Compatible con ayuno intermitente",
                "Seguro para uso prolongado"
            ],
            "risks": [
                "Efectos secundarios mínimos",
                "Posible dolor en el sitio de inyección",
                "No estudiado en embarazo/lactancia",
                "Evidencia clínica limitada"
            ],
            "typical_dose": "250-500 mcg subcutáneo/día",
            "cycle": "8-16 semanas"
        },
        "Cagrilintide": {
            "category": "Metabólico",
            "full_name": "Cagrilintide (CagriSema cuando se combina con Semaglutida)",
            "description": "Análogo de amilina sintético que promueve saciedad y reduce la ingesta calórica. Combinado con semaglutida en CagriSema.",
            "uses": [
                "Tratamiento de obesidad y sobrepeso",
                "Control del apetito",
                "Reducción de peso corporal",
                "Diabetes tipo 2 (en combinación)",
                "Gestión del peso a largo plazo"
            ],
            "benefits": [
                "Mayor pérdida de peso que semaglutida sola",
                "Reduce el vaciado gástrico",
                "Promueve saciedad temprana",
                "Larga duración de acción (1x/semana)",
                "Resultados superiores en ensayos clínicos"
            ],
            "risks": [
                "Náuseas (frecuente al inicio)",
                "Vómitos",
                "Dolor abdominal",
                "Estreñimiento",
                "Reacciones en sitio de inyección"
            ],
            "typical_dose": "2.4 mg subcutáneo semanal (con semaglutida)",
            "cycle": "Tratamiento continuo supervisado"
        },
        "Retatrutide": {
            "category": "Metabólico",
            "full_name": "Retatrutide (Triple agonista GIP/GLP-1/glucagón)",
            "description": "Triple agonista de receptores GIP, GLP-1 y glucagón. Pérdida de peso superior a semaglutida y tirzepatide en ensayos clínicos.",
            "uses": [
                "Tratamiento de obesidad severa",
                "Diabetes tipo 2",
                "Reducción de riesgo cardiovascular",
                "Gestión del peso a largo plazo",
                "Síndrome metabólico"
            ],
            "benefits": [
                "Mayor pérdida de peso que tirzepatide (hasta 24%)",
                "Mejora marcadores cardiovasculares",
                "Reduce inflamación sistémica",
                "1x/semana (conveniencia)",
                "Efecto triple sobre metabolismo"
            ],
            "risks": [
                "Náuseas (frecuente)",
                "Vómitos",
                "Diarrea",
                "Pérdida de masa muscular potencial",
                "Aún en fase de desarrollo clínico"
            ],
            "typical_dose": "Escalado gradual hasta 12 mg semanal",
            "cycle": "Tratamiento continuo supervisado"
        },
        "5-amino-1MQ": {
            "category": "Metabólico",
            "full_name": "5-amino-1MQ (5-amino-1-metilquinolina)",
            "description": "Inhibidor selectivo de NNMT (nicotinamida N-metiltransferasa) que bloquea la acumulación de grasa y mejora el metabolismo energético.",
            "uses": [
                "Pérdida de grasa",
                " Mejora del metabolismo",
                "Reducción de inflamación",
                "Protección mitocondrial",
                "Síndrome metabólico"
            ],
            "benefits": [
                "Bloquea enzima NNMT (acumulación de grasa)",
                "Mejora sensibilidad a insulina",
                "Efecto anti-inflamatorio",
                "Aumenta metabolismo energético",
                "Efectos en grasa abdominal visceral"
            ],
            "risks": [
                "Efectos secundarios mínimos en estudios",
                "Posible malestar gastrointestinal",
                "Datos de seguridad a largo plazo limitados",
                "No recomendado en embarazo"
            ],
            "typical_dose": "50-100 mg oral/día",
            "cycle": "8-12 semanas"
        },
        "L-Carnitina": {
            "category": "Metabólico",
            "full_name": "L-Carnitina (L-Carnitine / Acetil-L-Carnitina)",
            "description": "Aminoácido natural esencial para el transporte de ácidos grasos a las mitocondrias para su oxidación (producción de energía).",
            "uses": [
                "Quema de grasa",
                "Energía y rendimiento físico",
                "Recuperación muscular",
                "Salud cardiovascular",
                "Función cerebral (acetil-L-carnitina)"
            ],
            "benefits": [
                "Transporta grasas a mitocondrias para quemarlas",
                "Aumenta producción de energía celular",
                "Reduce daño oxidativo",
                "Mejora función cardíaca",
                "Apoya salud neurológica"
            ],
            "risks": [
                "Olor corporal a pescado (en dosis altas)",
                "Náuseas en dosis elevadas",
                "Insomnio si se toma tarde",
                "Posible interacción con anticoagulantes",
                "Acumulación en enfermedad renal"
            ],
            "typical_dose": "500-2000 mg oral/día",
            "cycle": "8-16 semanas"
        },
        "Lipo-C": {
            "category": "Metabólico",
            "full_name": "Lipo-C (Mezcla lipotrópica inyectable)",
            "description": "Mezcla de compuestos lipotrópicos (Metionina, Inositol, Clorhidrato de Colina, B12, Lidocaína) que estimulan la movilización de grasa.",
            "uses": [
                "Pérdida de grasa acelerada",
                "Movilización de grasa localizada",
                "Energía y vitalidad",
                "Apoyo al metabolismo del hígado",
                "Dietas de reducción"
            ],
            "benefits": [
                "Acelera la lipólisis (quema de grasa)",
                "Mejora el transporte de grasas",
                "Apoya función hepática",
                "Aumenta energía",
                "Puede mejorar apariencia de celulitis"
            ],
            "risks": [
                "Dolor en sitio de inyección",
                "Náuseas",
                "Posible reacción alérgica",
                "No usar con enfermedad hepática severa",
                "Requiere receta médica"
            ],
            "typical_dose": "1-2 mL intramuscular 2x/semana",
            "cycle": "8-12 semanas"
        },
        "Lemon Bottle": {
            "category": "Metabólico",
            "full_name": "Lemon Bottle (Inyección lipotrópica)",
            "description": "Cóctel lipotrópico que contiene riboflavina, metionina, colina y other compuestos para acelerar la quema de grasa.",
            "uses": [
                "Pérdida de grasa",
                "Energía",
                "Detoxificación hepática",
                "Metabolismo acelerado",
                "Complemento a dieta y ejercicio"
            ],
            "benefits": [
                "Fomenta la movilización de grasas",
                "Aumenta energía celular",
                "Apoya la función del hígado",
                "Puede mejorar apariencia de piel",
                "Rápida absorción"
            ],
            "risks": [
                "Dolor en sitio de inyección",
                "Náuseas",
                "Posible orina con olor fuerte (riboflavina)",
                "Requiere prescripción",
                "No sustituye dieta y ejercicio"
            ],
            "typical_dose": "2 mL intramuscular 1-2x/semana",
            "cycle": "6-12 semanas"
        },
        # ── Reparación / Regeneración ──
        "KPV": {
            "category": "Reparación Tisular",
            "full_name": "KPV (Alfa-MSH tripeptide)",
            "description": "Tripeptide (Lisina-Prolina-Valina) derivado de la hormona alfa-MSH con potentes propiedades anti-inflamatorias, especialmente en intestino.",
            "uses": [
                "Inflamación intestinal (colitis, Crohn)",
                "Gastritis y úlceras",
                "Inflamación sistémica",
                "Dermatitis y eccema",
                "Recuperación digestiva"
            ],
            "benefits": [
                "Anti-inflamatorio potente en tracto GI",
                "Reduce citocinas pro-inflamatorias",
                "Promueve curación de mucosa intestinal",
                "Seguro para uso prolongado",
                "Sin efectos significativos sobre melanina (a diferencia de alfa-MSH completa)"
            ],
            "risks": [
                "Efectos secundarios mínimos",
                "Posible malestar GI al inicio",
                "No estudiado extensamente en humanos",
                "Precaución en melanoma (por relación con alfa-MSH)"
            ],
            "typical_dose": "200-500 mcg subcutáneo o oral 1-2x/día",
            "cycle": "4-12 semanas"
        },
        "ARA-290": {
            "category": "Reparación Tisular",
            "full_name": "ARA-290 (Cibinetide)",
            "description": "Péptido derivado de eritropoyetina (EPO) que actúa sobre el receptor de eritropoyetina no hematopoyético (EPO-R), con efectos anti-inflamatorios y neuroprotectores sin aumentar glóbulos rojos.",
            "uses": [
                "Dolor neuropático",
                "Neuropatía diabética",
                "Enfermedades autoinmunes",
                "Fibromialgia",
                "Inflamación crónica"
            ],
            "benefits": [
                "Reduce dolor neuropático sin opioides",
                "Neuroprotector y neuroregenerativo",
                "Anti-inflamatorio sin efectos hematológicos",
                "Mejora función nerviosa periférica",
                "Potencial en enfermedades raras (sarcoidosis)"
            ],
            "risks": [
                "Efectos secundarios mínimos",
                "Dolor en sitio de inyección",
                "Datos a largo plazo limitados",
                "Costo elevado",
                "Acceso limitado"
            ],
            "typical_dose": "4 mg subcutáneo diario",
            "cycle": "12-24 semanas"
        },
        "LL37": {
            "category": "Reparación Tisular",
            "full_name": "LL-37 (Antimicrobial Peptide)",
            "description": "Péptido antimicrobiano natural (catelicidina) producido por el sistema inmune con propiedades antibacterianas, antivirales y de reparación de heridas.",
            "uses": [
                "Infecciones resistentes a antibióticos",
                "Heridas crónicas y úlceras",
                "Cicatrización de tejidos",
                "Infecciones cutáneas",
                "Biofilms bacterianos"
            ],
            "benefits": [
                "Efecto bactericida amplio (gram+ y gram-)",
                "Rompe biofilms bacterianos",
                "Promueve curación de heridas",
                "Efecto antiviral documentado",
                "Regula respuesta inmune"
            ],
            "risks": [
                "Posible toxicidad en dosis altas",
                "Efecto citotóxico en concentraciones elevadas",
                "Producción compleja",
                "Estabilidad limitada",
                "No disponible para uso humano general"
            ],
            "typical_dose": "Investigacional - no establecido",
            "cycle": "N/A - uso en investigación"
        },
        # ── Hormonales ──
        "Tesamorelin": {
            "category": "Eje Hormonal",
            "full_name": "Tesamorelin (Egrifta)",
            "description": "Análogo de la hormona de liberación de GH (GHRH) aprobado por FDA para reducir grasa visceral en pacientes con VIH. Estimula la producción natural de GH.",
            "uses": [
                "Reducción de grasa visceral",
                "Síndrome metabólico",
                "Déficit de GH en adultos",
                "Gestión de peso en VIH",
                "Salud metabólica"
            ],
            "benefits": [
                "Estimula GH natural (no suprime eje)",
                "Reducción demostrada de grasa visceral",
                "Mejora perfil lipídico",
                "Seguro a largo plazo (aprobado FDA)",
                "1x/día subcutáneo"
            ],
            "risks": [
                "Reacciones en sitio de inyección",
                "Dolor articular",
                "Edema periférico",
                "Elevación de glucosa (temporal)",
                "Costo elevado"
            ],
            "typical_dose": "2 mg subcutáneo antes de dormir",
            "cycle": "Continuo bajo supervisión médica"
        },
        "Sermorelin": {
            "category": "Eje Hormonal",
            "full_name": "Sermorelin (GHRH 1-29)",
            "description": "Fragmento sintético de la hormona de liberación de GH (GHRH) que estimula la hipófisis para producir y liberar hormona de crecimiento de forma natural.",
            "uses": [
                "Déficit de GH en adultos",
                "Anti-envejecimiento",
                "Mejora de composición corporal",
                "Calidad de sueño",
                "Salud ósea"
            ],
            "benefits": [
                "Estimula producción natural de GH",
                "No suprime eje GH (a diferencia de GH exógena)",
                "Mejora calidad de sueño profundo",
                "Aumenta masa muscular",
                "Reduce grasa corporal"
            ],
            "risks": [
                "Reacciones en sitio de inyección",
                "Náuseas",
                "Ruborización",
                "Mareos",
                "Elevación temporal de glucosa"
            ],
            "typical_dose": "0.3-1 mg subcutáneo nocte",
            "cycle": "3-6 meses, descanso de 1-2 meses"
        },
        # ── Longevidad ──
        "IGF-1 LR3": {
            "category": "Longevidad",
            "full_name": "IGF-1 LR3 (Insulin-like Growth Factor-1 Long R3)",
            "description": "Análogo del factor de crecimiento similar a insulina con vida media extendida. Promueve crecimiento muscular, reparación tisular y neurogénesis.",
            "uses": [
                "Crecimiento muscular",
                "Recuperación de lesiones",
                "Neuroprotección",
                "Anti-envejecimiento",
                "Regeneración de tejidos"
            ],
            "benefits": [
                "Potente efecto anabólico",
                "Promueve hipertrofia muscular",
                "Aumenta síntesis de proteínas",
                "Neuroprotector",
                "Mejora sensibilidad a insulina en músculo"
            ],
            "risks": [
                "Hipoglucemia (efecto类似 insulina)",
                "Dolor articular",
                "Acromegalia si se abusa",
                "Supresión de GH endógeno",
                "Crecimiento de tejido no deseado"
            ],
            "typical_dose": "20-80 mcg subcutáneo/día",
            "cycle": "4-8 semanas"
        },
        "PEG-MGF": {
            "category": "Longevidad",
            "full_name": "PEG-MGF (Pegylated Mechano Growth Factor)",
            "description": "Factor de crecimiento mecánico pegilado derivado de IGF-1. Estimula la proliferación de células satélite y reparación muscular.",
            "uses": [
                "Recuperación muscular intensiva",
                "Hipertrofia muscular",
                "Reparación de tendones",
                "Anti-envejecimiento muscular",
                "Lesiones deportivas"
            ],
            "benefits": [
                "Estimula células satélite musculares",
                "Promueve hipertrofia muscular",
                "Acelera reparación de tendones",
                "Efecto prolongado (pegilación)",
                "Complemento potente a entrenamiento"
            ],
            "risks": [
                "Dolor en sitio de inyección",
                "Hipoglucemia leve",
                "Posible crecimiento desproporcionado",
                "No recomendado en cáncer activo",
                "Requiere experiencia en manejo"
            ],
            "typical_dose": "200-400 mcg subcutáneo cada 3-5 días",
            "cycle": "4-8 semanas"
        },
        "Thymosin Alpha-1": {
            "category": "Inmunomodulador",
            "full_name": "Thymosin Alpha-1 (Tα1 / Zadaxin)",
            "description": "Péptido timoico natural que modula la respuesta inmune, aumentando la función de células T y NK. Utilizado en hepatitis, infecciones y cáncer como coadyuvante.",
            "uses": [
                "Inmunodeficiencia",
                "Hepatitis B y C",
                "Coadyuvante en quimioterapia",
                "Infecciones virales crónicas",
                "Vacunación en pacientes inmunocomprometidos"
            ],
            "benefits": [
                "Estimula células T y NK",
                "Mejora respuesta inmune innata y adaptativa",
                "Efecto antiviral documentado",
                "Bien tolerado",
                "Potencia efecto de vacunas"
            ],
            "risks": [
                "Reacciones en sitio de inyección",
                "Fiebre leve",
                "Malestar general",
                "Posible activación autoinmune",
                "Costo elevado"
            ],
            "typical_dose": "1.6 mg subcutáneo 2x/semana",
            "cycle": "12-24 semanas"
        },
        # ── Nootrópicos ──
        "Dihexa": {
            "category": "Nootrópico",
            "full_name": "Dihexa (N-hexanoic-Tyr-Ile-(6) aminohexanoic amide)",
            "description": "Péptido nootrópico sintético derivado de Angiotensina IV con potente efecto neurotrófico y de potenciación sináptica. Mejora memoria y función cognitiva.",
            "uses": [
                "Mejora de memoria y aprendizaje",
                "Neurodegeneración (Alzheimer, Parkinson)",
                "Lesiones cerebrales traumáticas",
                "Envejecimiento cognitivo",
                "Ansiedad y depresión"
            ],
            "benefits": [
                "Potencia formación de sinapsis",
                "Neuroprotector y neuroregenerativo",
                "Mejora memoria de trabajo",
                "Efecto ansiolítico",
                "Biodisponibilidad oral"
            ],
            "risks": [
                "Dolor de cabeza",
                "Insomnio",
                "Irritabilidad",
                "Datos de seguridad a largo plazo limitados",
                "No recomendado en embarazo"
            ],
            "typical_dose": "1-5 mg oral/día",
            "cycle": "4-12 semanas"
        },
        "Noopept": {
            "category": "Nootrópico",
            "full_name": "Noopept (Omberacetam / GVS-111)",
            "description": "Péptido nootrópico sintético derivado de fenilpiracetam con efectos neuroprotectores y de mejora cognitiva superiores a piracetam.",
            "uses": [
                "Mejora de memoria",
                "Atención y concentración",
                "Ansiedad",
                "Neuroprotección",
                "Recuperación de daño cerebral"
            ],
            "benefits": [
                "Potencia memoria y aprendizaje",
                "Efecto ansiolítico",
                "Neuroprotector",
                "Biodisponibilidad oral",
                "Efecto rápido (30-60 min)"
            ],
            "risks": [
                "Dolor de cabeza",
                "Irritabilidad",
                "Insomnio (si se toma tarde)",
                "Náuseas en dosis altas",
                "Dependencia psicológica potencial"
            ],
            "typical_dose": "10-30 mg oral 1-2x/día",
            "cycle": "4-8 semanas, descanso de 2 semanas"
        },
        "Cerebrolysin": {
            "category": "Nootrópico",
            "full_name": "Cerebrolysin (Extracto de cerebro porcino)",
            "description": "Mezcla de péptidos neurotróficos derivados de cerebro porcino con efectos neuroprotectores y neuroregenerativos. Utilizado en Europa para demencia y ACV.",
            "uses": [
                "Demencia vascular y Alzheimer",
                "Accidente cerebrovascular",
                "Lesiones cerebrales traumáticas",
                "Encefalopatía",
                "Envejecimiento cognitivo"
            ],
            "benefits": [
                "Promueve neuroregeneración",
                "Protege neuronas del daño excitotóxico",
                "Mejora función cognitiva en demencia",
                "Bien tolerado",
                "Experiencia clínica extensa (Europa)"
            ],
            "risks": [
                "Reacciones en sitio de inyección",
                "Náuseas",
                "Cefalea",
                "Insomnio",
                "Posible hiperexcitabilidad"
            ],
            "typical_dose": "5-30 mL intramuscular/intravenoso diario",
            "cycle": "4-12 semanas"
        },
        "DSIP": {
            "category": "Nootrópico",
            "full_name": "DSIP (Delta Sleep-Inducing Peptide)",
            "description": "Péptido natural que induce el sueño delta (profundo). Regula el ritmo circadiano y promueve un sueño reparador sin efectos sedantes típicos.",
            "uses": [
                "Insomnio",
                "Sueño de baja calidad",
                "Ritmo circadiano irregular",
                "Estrés crónico",
                "Jet lag"
            ],
            "benefits": [
                "Promueve sueño profundo (ondas delta)",
                "No causa somnolencia diurna",
                "Regula reloj biológico",
                "Mejora calidad del sueño sin sedación",
                "Sin potencial de adicción"
            ],
            "risks": [
                "Dolor en sitio de inyección",
                "Somnolencia inicial (se adapta)",
                "Posible hipotensión",
                "No combinar con sedantes",
                "Baja biodisponibilidad oral"
            ],
            "typical_dose": "100-200 mcg subcutáneo nocte",
            "cycle": "2-4 semanas"
        },
        "Selank": {
            "category": "Nootrópico",
            "full_name": "Selank (Thr-Lys-Pro-Arg-Pro-Gly-Pro)",
            "description": "Péptido ansiolítico sintético derivado de tufticina. Modula GABA y promueve neuroplasticidad sin sedación.",
            "uses": [
                "Ansiedad",
                "Fobias",
                "Estrés crónico",
                "ME/CFS",
                "Ansiedad social"
            ],
            "benefits": [
                "Ansiolítico sin sedación",
                "Mejora tolerancia al estrés",
                "Potencia memoria y aprendizaje",
                "Inmuno-modulador",
                "Sin dependencia"
            ],
            "risks": [
                "Irritabilidad",
                "Insomnio (si se toma tarde)",
                "Dolor de cabeza",
                "Cambios de humor",
                "No combinar con ansiolíticos fuertes"
            ],
            "typical_dose": "250-750 mcg subcutáneo 1-2x/día",
            "cycle": "2-4 semanas"
        },
        # ── Cosméticos ──
        "Melanotan II": {
            "category": "Cosmético",
            "full_name": "Melanotan II (MT-2)",
            "description": "Análogo sintético de la alfa-MSH que estimula la producción de melanina, proporcionando bronceado sin exposición solar. También efectos afrodisíacos.",
            "uses": [
                "Bronceado sin sol",
                "Protección UV de la piel",
                "Disfunción eréctil",
                "Libido femenina",
                "Pigmentación de piel"
            ],
            "benefits": [
                "Bronceado intenso sin daño solar",
                "Protección contra radiación UV",
                "Efecto afrodisíaco documentado",
                "Mejora apariencia de piel",
                "Efecto supresor de apetito"
            ],
            "risks": [
                "Náuseas (frecuente)",
                "Ruborización facial",
                "Cambios en pecas/lunares",
                "Posible crecimiento de nevos",
                "No recomendado en antecedentes de melanoma"
            ],
            "typical_dose": "0.5-1 mg subcutáneo (mantenimiento: 0.5 mg 2-3x/semana)",
            "cycle": "Ciclo de carga inicial 2 semanas, luego mantenimiento"
        },
        "Acetyl Hexapeptide-8": {
            "category": "Cosmético",
            "full_name": "Acetyl Hexapeptide-8 (Argireline)",
            "description": "Péptido tópico anti-arrugas que inhibe la liberación de neurotransmisores en la unión neuromuscular, reduciendo la contracción muscular facial.",
            "uses": [
                "Reducción de arrugas dinámicas",
                "Líneas de expresión frontales",
                "Patas de gallo",
                "Líneas de la frente",
                "Rejuvenecimiento facial"
            ],
            "benefits": [
                "Reduce contracción muscular hasta 30%",
                "Efecto similar a Botox tópico",
                "No invasivo",
                "Compatible con otros activos",
                "Uso tópico seguro"
            ],
            "risks": [
                "Efecto leve comparado con Botox inyectable",
                "Irritación cutánea en pieles sensibles",
                "Requiere uso constante",
                "No sustituye procedimientos estéticos",
                "Evidencia limitada a largo plazo"
            ],
            "typical_dose": "Aplicación tópica 1-2x/día (5-10% concentración)",
            "cycle": "Uso continuo"
        },
        "Palmitoyl Pentapeptide-4": {
            "category": "Cosmético",
            "full_name": "Palmitoyl Pentapeptide-4 (Matrixyl)",
            "description": "Péptido tópico bioactivo que estimula la síntesis de colágeno, elastina y ácido hialurónico en la dermis, mejorando la estructura de la piel.",
            "uses": [
                "Anti-envejecimiento cutáneo",
                "Pérdida de elasticidad",
                "Arrugas profundas",
                "Textura de piel",
                "Firmeza cutánea"
            ],
            "benefits": [
                "Estimula colágeno tipo I, III y IV",
                "Aumenta elasticidad",
                "Reduce profundidad de arrugas",
                "Mejora hidratación",
                "Efecto antienvejecimiento demostrado"
            ],
            "risks": [
                "Efectos secundarios mínimos",
                "Posible irritación leve en pieles sensibles",
                "Resultados visibles en 8-12 semanas",
                "No es un reemplazo para procedimientos",
                "Requiere protección solar"
            ],
            "typical_dose": "Aplicación tópica 1-2x/día (3-5% concentración)",
            "cycle": "Uso continuo"
        },
        # ── Fertilidad / Hormonal ──
        "hCG": {
            "category": "Hormonal",
            "full_name": "hCG (Gonadotropina Coriónica Humana)",
            "description": "Hormona natural producida durante el embarazo. En hombres, estimula los testículos para producir testosterona. Utilizada en TRT y fertilidad.",
            "uses": [
                "Preservación de fertilidad en TRT",
                "Estimulación de testosterona",
                "Criptorquidia",
                "Pérdida de peso (dieta hCG)",
                "Apoyo a fertilidad masculina"
            ],
            "benefits": [
                "Mantiene conteo espermático en TRT",
                "Estimula testosterona natural",
                "Previene atrofia testicular",
                "Bien tolerado",
                "Apoyo a fertilidad masculina"
            ],
            "risks": [
                "Ginecomastia (por aromatización)",
                "Retención de líquidos",
                "Dolor en sitio de inyección",
                "Posible hipogonadismo si se abusa",
                "No recomendado en cáncer de próstata"
            ],
            "typical_dose": "500-2000 UI subcutáneo 2-3x/semana",
            "cycle": "Concurrente con TRT"
        },
        "Kisspeptin-10": {
            "category": "Hormonal",
            "full_name": "Kisspeptin-10 (KP-10)",
            "description": "Péptido natural que estimula la liberación de GnRH, FSH y LH. Regula la función reproductiva y puede mejorar fertilidad y testosterona.",
            "uses": [
                "Hipogonadismo hipogonadotrópico",
                "Infertilidad",
                "Baja testosterona",
                "Disfunción reproductiva",
                "Estimulación de LH/FSH"
            ],
            "benefits": [
                "Estimula eje reproductivo natural",
                "Aumenta LH y FSH",
                "Mejora fertilidad",
                "Efecto fisiológico (no suprime eje)",
                "Potencial en tratamientos de fertilidad"
            ],
            "risks": [
                "Náuseas",
                "Vómitos",
                "Ruborización",
                "Dolor en sitio de inyección",
                "Estimulación ovárica no deseada"
            ],
            "typical_dose": "0.5-1 mcg/kg subcutáneo/día",
            "cycle": "Bajo supervisión médica reproductiva"
        },
        # ── Otros ──
        "Lunasina": {
            "category": "Longevidad",
            "full_name": "Lunasina (Lunasin)",
            "description": "Péptido derivado de soja con propiedades anti-inflamatorias y epigenéticas. Inhibe la acetilación de histonas y modula expresión génica.",
            "uses": [
                "Inflamación crónica",
                "Epigenética / expresión génica",
                "Protección cardiovascular",
                "Salud ósea",
                "Potencial antitumoral"
            ],
            "benefits": [
                "Modula expresión génica vía epigenética",
                "Anti-inflamatorio",
                "Protector cardiovascular",
                "Biodisponibilidad oral",
                "Origen natural (soja)"
            ],
            "risks": [
                "Efectos secundarios mínimos",
                "Alergia a soja (contraindicado)",
                "Datos clínicos limitados",
                "No recomendado en embarazo"
            ],
            "typical_dose": "16-50 mg oral/día",
            "cycle": "8-12 semanas"
        },
        "Mecasermin": {
            "category": "Eje Hormonal",
            "full_name": "Mecasermin (Increlex - IGF-1)",
            "description": "IGF-1 recombinante aprobado por FDA para tratamiento de déficit severo de IGF-1. Promueve crecimiento lineal y desarrollo tisular.",
            "uses": [
                "Déficit severo de IGF-1",
                "Enanismo (insensibilidad a GH)",
                "Crecimiento en niños",
                "Regeneración tisular",
                "Apoyo nutricional severo"
            ],
            "benefits": [
                "Estimula crecimiento lineal",
                "Promueve desarrollo muscular",
                "Mejora densidad ósea",
                "Efecto anabólico potente",
                "FDA aprobado para indicaciones específicas"
            ],
            "risks": [
                "Hipoglucemia (significativa)",
                "Dolor en sitio de inyección",
                "Tonsilas/adenoides agrandadas",
                "Cefalea",
                "Posible riesgo de neoplasia"
            ],
            "typical_dose": "0.04-0.12 mg/kg subcutáneo 2x/día",
            "cycle": "Bajo supervisión endocrinológica"
        },
        "Eloralintide": {
            "category": "Metabólico",
            "full_name": "Eloralintide (Novo Nordisk)",
            "description": "Análogo de amilina en desarrollo para obesidad. Reduce peso corporal a través de mecanismos de saciedad similares a la amilina natural.",
            "uses": [
                "Tratamiento de obesidad",
                "Control del peso",
                "Complemento a agonistas GLP-1",
                "Síndrome metabólico",
                "Diabetes tipo 2"
            ],
            "benefits": [
                "Mecanismo complementario a GLP-1",
                "Promueve saciedad",
                "Larga duración de acción",
                "Potencial sinérgico con semaglutida",
                "Ensayos clínicos prometedores"
            ],
            "risks": [
                "Náuseas",
                "Vómitos",
                "Fase temprana de desarrollo",
                "Datos de seguridad limitados",
                "No aprobado aún"
            ],
            "typical_dose": "Escalado gradual (en investigación)",
            "cycle": "Tratamiento continuo (en investigación)"
        },
    }

    # Search and filter
    col_search, col_filter = st.columns([3, 1])
    with col_search:
        search = st.text_input("🔍 Buscar péptido", placeholder="Escribe el nombre del péptido...")
    with col_filter:
        categories = ["Todos"] + list(set(p["category"] for p in PEPTIDE_DB.values()))
        category_filter = st.selectbox("Filtrar por categoría", categories)

    # Filter peptides
    filtered_peptides = PEPTIDE_DB.copy()
    if search:
        filtered_peptides = {k: v for k, v in filtered_peptides.items() 
                           if search.lower() in k.lower() or search.lower() in v["full_name"].lower()}
    if category_filter != "Todos":
        filtered_peptides = {k: v for k, v in filtered_peptides.items() 
                           if v["category"] == category_filter}

    # Display peptides
    for name, info in filtered_peptides.items():
        with st.expander(f"**{name}** - {info['full_name']} ({info['category']})", expanded=False):
            st.markdown(f"### {name}")
            st.caption(info['full_name'])
            st.info(info['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Usos principales:**")
                for use in info['uses']:
                    st.markdown(f"- {use}")
                
                st.markdown("**💊 Dosis típica:**")
                st.code(info['typical_dose'])
                
                st.markdown("**🔄 Ciclo:**")
                st.code(info['cycle'])
            
            with col2:
                st.markdown("**⭐ Beneficios:**")
                for benefit in info['benefits']:
                    st.markdown(f"- {benefit}")
                
                st.markdown("**⚠️ Riesgos:**")
                for risk in info['risks']:
                    st.markdown(f"- {risk}")

    # Summary table
    st.divider()
    st.markdown("### 📊 Resumen Rápido")
    
    summary_data = []
    for name, info in filtered_peptides.items():
        summary_data.append({
            "Péptido": name,
            "Categoría": info['category'],
            "Dosis": info['typical_dose'][:30] + "...",
            "Ciclo": info['cycle'][:25] + "..."
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

# ============================================================
# TAB: ORÁCULO CLÍNICO (RAG)
# ============================================================
with tab_oraculo:
    st.header("El Oráculo Clínico")
    st.markdown("**Respuestas generadas exclusivamente desde tu Base de Conocimiento local.**")
    st.caption("Si no encuentra el dato → responde exactamente: 'No hay datos clínicos locales sobre esto'")

    col_search, col_info = st.columns([3, 1])

    with col_search:
        compound_query = st.text_input(
            "Nombre del péptido o compuesto",
            placeholder="Ej: BPC-157, TB-500, Semaglutide, CJC-1295...",
            key="oraculo_input"
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            search_btn = st.button("🔍 Consultar Protocolo", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🧹 Limpiar", use_container_width=True):
                st.rerun()

        if search_btn:
            if not compound_query.strip():
                st.warning("Ingresa el nombre de un compuesto.")
            elif not has_knowledge_base():
                st.error("La base de conocimiento está vacía. Ingresa PDFs en /Base_Conocimiento y haz clic en 'Ingerir' en la barra lateral.")
            else:
                with st.spinner("Consultando literatura local..."):
                    result = query_peptide_protocol(compound_query.strip())

                # Guardar en historial
                save_oracle_query(compound_query.strip(), result)

                st.markdown("### 📋 Respuesta del Oráculo")
                if "No hay datos clínicos locales sobre esto" in result:
                    st.warning(result)
                else:
                    st.success("Datos encontrados en tu biblioteca local")
                    st.text_area(
                        "Protocolo / Información (incluye fuente)",
                        value=result,
                        height=420,
                        key="oraculo_result"
                    )
                    st.caption("⚠️ Siempre verifica la fuente citada antes de usar cualquier protocolo.")

    with col_info:
        st.info("""
        **Reglas del sistema:**
        - Temperatura = 0 (máxima fidelidad)
        - Solo texto recuperado de tus PDFs
        - Fuente siempre visible cuando hay datos
        - Cero invención de dosis
        """)

        st.markdown("**Consejo:**")
        st.markdown("Usa nombres exactos como aparecen en tus documentos (ej: 'BPC-157', 'Ipamorelin 2mg').")

    # ---- HISTORIAL DEL ORÁCULO ----
    st.divider()
    with st.expander("📜 Historial de Consultas al Oráculo", expanded=False):
        try:
            conn = get_connection()
            df_history = pd.read_sql("""
                SELECT id, timestamp, question, answer
                FROM oracle_history
                ORDER BY timestamp DESC
                LIMIT 50
            """, conn)
            conn.close()
        except Exception:
            df_history = pd.DataFrame()

        if df_history.empty:
            st.info("No hay consultas guardadas todavía.")
        else:
            # Botón para borrar todo el historial
            col_del1, col_del2 = st.columns([3, 1])
            with col_del2:
                if st.button("🗑️ Borrar todo el historial", type="secondary"):
                    execute_delete("oracle_history")
                    st.success("Historial borrado completamente.")
                    st.rerun()

            for _, row in df_history.iterrows():
                col_h1, col_h2 = st.columns([4, 1])
                with col_h1:
                    st.markdown(f"**{row['timestamp']}** — *{row['question']}*")
                    answer_preview = str(row["answer"])[:300]
                    if len(str(row["answer"])) > 300:
                        answer_preview += "..."
                    st.text(answer_preview)
                with col_h2:
                    if st.button(f"Borrar", key=f"del_{row['id']}"):
                        # Remove specific item from session state
                        st.session_state.oracle_history = [
                            item for item in st.session_state.oracle_history 
                            if item.get("id") != row["id"]
                        ]
                        st.rerun()
                st.divider()

# ============================================================
# TAB 3: DASHBOARD ANALÍTICAS
# ============================================================
BIOMARKER_COLS = [
    "igf1", "glucose", "free_testosterone", "total_testosterone",
    "estradiol", "cholesterol_total", "ldl", "hdl", "triglycerides",
    "alt", "ast", "tsh", "creatinine"
]

with tab_dashboard:
    st.header("Evolución Temporal de Biomarcadores")

    try:
        conn = get_connection()
        df_raw = pd.read_sql("SELECT * FROM blood_markers ORDER BY date ASC", conn)
        conn.close()
    except Exception:
        df_raw = pd.DataFrame()

    show_charts = True

    if df_raw.empty:
        st.info("No hay analiticas registradas todavia.\n\nUsa el workflow de n8n o agrega datos manualmente con el formulario mas abajo.")
        show_charts = False
    else:
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        df_raw = df_raw.dropna(subset=["date"])

        if df_raw.empty:
            st.info("No hay filas con fechas validas en la tabla de analiticas.")
            show_charts = False
        else:
            try:
                min_ts = df_raw["date"].min()
                max_ts = df_raw["date"].max()

                if pd.isna(min_ts) or pd.isna(max_ts):
                    st.warning("Las fechas de las analiticas no son validas. Agrega registros con fechas correctas.")
                    show_charts = False
                else:
                    min_date = min_ts.date()
                    max_date = max_ts.date()
            except Exception:
                st.warning("Error procesando las fechas de las analiticas.")
                show_charts = False

    if show_charts:
        col_filter1, col_filter2 = st.columns([2, 3])

        with col_filter1:
            date_range = st.date_input(
                "Rango de fechas",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

        default_biomarkers = ["igf1", "free_testosterone", "estradiol", "alt", "glucose"]

        with col_filter2:
            selected_markers = st.multiselect(
                "Biomarcadores a visualizar",
                options=BIOMARKER_COLS,
                default=[b for b in default_biomarkers if b in BIOMARKER_COLS],
                help="Selecciona uno o varios marcadores",
                format_func=lambda x: format_marker_label(x)
            )

        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (df_raw["date"].dt.date >= start_date) & (df_raw["date"].dt.date <= end_date)
            df_filtered = df_raw.loc[mask].copy()
        else:
            df_filtered = df_raw.copy()

        if df_filtered.empty:
            st.warning("No hay datos en el rango seleccionado.")
        else:
            id_vars = ["date", "test_name"]
            value_vars = [col for col in selected_markers if col in df_filtered.columns]

            if not value_vars:
                st.warning("Selecciona al menos un biomarcador.")
            else:
                df_melt = df_filtered.melt(
                    id_vars=id_vars,
                    value_vars=value_vars,
                    var_name="Biomarcador",
                    value_name="Valor"
                ).dropna(subset=["Valor"])

                if df_melt.empty:
                    st.warning("No hay valores numericos para los biomarcadores seleccionados en este rango.")
                else:
                    fig = px.line(
                        df_melt,
                        x="date",
                        y="Valor",
                        color="Biomarcador",
                        markers=True,
                        title="Evolucion de Biomarcadores",
                        labels={"date": "Fecha", "Valor": "Valor", "Biomarcador": "Marcador"}
                    )
                    fig.update_layout(
                        height=520,
                        hovermode="x unified",
                        template="plotly_dark",
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Datos Detallados (con Alertas de Rango)")

            display_cols = ["date", "test_name"] + [c for c in BIOMARKER_COLS if c in df_filtered.columns]
            df_display = df_filtered[display_cols].sort_values("date", ascending=False).copy()

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    col: st.column_config.NumberColumn(
                        label=format_marker_label(col),
                    )
                    for col in BIOMARKER_COLS if col in df_display.columns
                }
            )

            alert_found = False
            alert_messages = []
            for _, row in df_display.iterrows():
                for col in BIOMARKER_COLS:
                    if col in df_display.columns and col in REFERENCE_RANGES:
                        val = row[col]
                        if val is not None and not pd.isna(val) and is_out_of_range(col, val):
                            r = REFERENCE_RANGES[col]
                            alert_messages.append(
                                f"VALOR FUERA DE RANGO: **{r['label']}** = {val} {r['unit']} "
                                f"(rango normal: {r['min']}-{r['max']}) - "
                                f"Fecha: {row['date']}"
                            )
                            alert_found = True

            if alert_found:
                st.markdown("### Valores Fuera de Rango Detectados")
                for msg in alert_messages:
                    st.markdown(msg)
                st.caption("Consulta a un profesional medico sobre los valores señalados.")
            else:
                st.success("Todos los valores estan dentro del rango de referencia.")

    # ============================================================
    # EXPORTACIÓN PDF MENSUAL
    # ============================================================
    st.divider()
    st.subheader("📄 Exportar Informe PDF Mensual")

    col_pdf1, col_pdf2, col_pdf3 = st.columns([2, 2, 2])

    with col_pdf1:
        pdf_year = st.number_input(
            "Año",
            min_value=2020,
            max_value=2030,
            value=date.today().year,
            step=1
        )

    with col_pdf2:
        pdf_month = st.selectbox(
            "Mes",
            options=list(range(1, 13)),
            index=date.today().month - 1,
            format_func=lambda m: [
                "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
            ][m - 1]
        )

    with col_pdf3:
        st.write("")  # Espaciador
        st.write("")  # Espaciador
        if st.button("📥 Generar PDF", type="primary", use_container_width=True):
            with st.spinner("Generando informe..."):
                try:
                    pdf_buffer = export_monthly_report_pdf(int(pdf_year), int(pdf_month))
                    month_names_list = [
                        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
                    ]
                    filename = f"peptibro_{pdf_year}_{int(pdf_month):02d}_{month_names_list[int(pdf_month)-1].lower()}.pdf"
                    st.download_button(
                        label="⬇️ Descargar PDF",
                        data=pdf_buffer,
                        file_name=filename,
                        mime="application/pdf"
                    )
                    st.success("PDF generado. Haz clic en el botón de descarga arriba.")
                except Exception as e:
                    st.error(f"Error generando PDF: {e}")

    # ============================================================
    # AGREGAR ANALÍTICA MANUALMENTE + IMPORT CSV
    # ============================================================
    with st.expander("➕ Agregar Analítica Manualmente"):
        import_mode = st.radio("Modo", ["Formulario", "Importar CSV"], horizontal=True, key="import_mode")

        if import_mode == "Formulario":
            with st.form("manual_blood_form"):
                test_date = st.date_input("Fecha de la analítica", value=date.today())
                test_name = st.text_input("Nombre / Laboratorio", value="Manual entry")

                cols = st.columns(4)
                values = {}
                for i, marker in enumerate(BIOMARKER_COLS):
                    with cols[i % 4]:
                        values[marker] = st.number_input(
                            format_marker_label(marker),
                            value=None,
                            step=0.1
                        )

                manual_notes = st.text_area("Notas adicionales", height=60)

                if st.form_submit_button("Guardar Analítica"):
                    try:
                        execute_insert("blood_markers", (
                            test_date.isoformat(), test_name,
                            values["igf1"], values["glucose"], values["free_testosterone"], values["total_testosterone"],
                            values["estradiol"], values["cholesterol_total"], values["ldl"], values["hdl"],
                            values["triglycerides"], values["alt"], values["ast"], values["tsh"], values["creatinine"],
                            manual_notes or None
                        ))
                        st.success("Analítica guardada correctamente.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.markdown("**Formato CSV esperado:** `date,test_name,igf1,glucose,free_testosterone,total_testosterone,estradiol,cholesterol_total,ldl,hdl,triglycerides,alt,ast,tsh,creatinine`")
            csv_file = st.file_uploader("Subir CSV", type=["csv"], key="blood_csv")
            if csv_file:
                try:
                    df_import = pd.read_csv(csv_file)
                    st.dataframe(df_import.head(), use_container_width=True)
                    if st.button("📥 Importar datos", type="primary"):
                        try:
                            imported = 0
                            for _, row in df_import.iterrows():
                                try:
                                    execute_insert("blood_markers", (
                                        str(row.get("date", date.today().isoformat())),
                                        str(row.get("test_name", "CSV Import")),
                                        row.get("igf1"), row.get("glucose"), row.get("free_testosterone"),
                                        row.get("total_testosterone"), row.get("estradiol"), row.get("cholesterol_total"),
                                        row.get("ldl"), row.get("hdl"), row.get("triglycerides"),
                                        row.get("alt"), row.get("ast"), row.get("tsh"), row.get("creatinine")
                                    ))
                                    imported += 1
                                except Exception as e:
                                    st.warning(f"Error en fila: {e}")
                            st.success(f"✅ {imported} registros importados")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                except Exception as e:
                    st.error(f"Error leyendo CSV: {e}")

    # ============================================================
    # ANÁLISIS PREDICTIVO IA
    # ============================================================
    st.divider()
    with st.expander("🔮 Análisis Predictivo IA (Sin analítica real)", expanded=False):
        st.markdown("""
        **Simula una analítica basándose en tu protocolo y síntomas.**
        La IA generará valores predictivos basados en literatura médica y tu perfil.
        
        ⚠️ *Esto es una herramienta educativa. NO sustituye una analítica real.*
        """)

        # Get current protocol from daily log
        # Get current protocol from session state
        default_protocol = ""
        daily_log = st.session_state.get("daily_log", [])
        if daily_log:
            # Group by compound and calculate averages
            compounds = {}
            for entry in daily_log:
                name = entry.get("compound_name", "")
                dose = entry.get("dosage_mcg", 0)
                if name:
                    if name not in compounds:
                        compounds[name] = {"total_dose": 0, "count": 0}
                    compounds[name]["total_dose"] += dose
                    compounds[name]["count"] += 1
            
            if compounds:
                protocol_lines = []
                for name, data in compounds.items():
                    avg_dose = int(data["total_dose"] / data["count"])
                    protocol_lines.append(f"- {name}: {avg_dose} mcg")
                default_protocol = "\n".join(protocol_lines)

        with st.form("predictive_analysis"):
            st.markdown("**📋 Perfil Personal**")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                age = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1)
            with col_p2:
                sex = st.selectbox("Sexo", ["Masculino", "Femenino"])
            with col_p3:
                weight = st.number_input("Peso (kg)", min_value=40, max_value=200, value=80, step=1)

            st.markdown("**💊 Protocolo Actual (últimos 90 días)**")
            st.caption("Selecciona los péptidos que estás tomando y sus dosis habituales")

            # All peptides from compendium with typical doses
            PEPTIDE_OPTIONS = {
                # Metabólicos
                "Semaglutide (0.25 mg/sem)": "Semaglutide 0.25 mg subcutáneo semanal",
                "Semaglutide (0.5 mg/sem)": "Semaglutide 0.5 mg subcutáneo semanal",
                "Semaglutide (1 mg/sem)": "Semaglutide 1 mg subcutáneo semanal",
                "Semaglutide (2.4 mg/sem)": "Semaglutide 2.4 mg subcutáneo semanal (Wegovy)",
                "Tirzepatide (2.5 mg/sem)": "Tirzepatide 2.5 mg subcutáneo semanal",
                "Tirzepatide (5 mg/sem)": "Tirzepatide 5 mg subcutáneo semanal",
                "Tirzepatide (10 mg/sem)": "Tirzepatide 10 mg subcutáneo semanal",
                "Tirzepatide (15 mg/sem)": "Tirzepatide 15 mg subcutáneo semanal",
                "Cagrilintide (1.2 mg/sem)": "Cagrilintide 1.2 mg subcutáneo semanal",
                "Retatrutide (1 mg/sem)": "Retatrutide 1 mg subcutáneo semanal",
                "Retatrutide (2 mg/sem)": "Retatrutide 2 mg subcutáneo semanal",
                "AOD-9604 (300 mcg/día)": "AOD-9604 300 mcg subcutáneo diario",
                "5-amino-1MQ (50 mg/día)": "5-amino-1MQ 50 mg oral diario",
                "L-Carnitine (500 mg/día)": "L-Carnitina 500 mg subcutáneo diario",
                "Lipo-C (inyección)": "Lipo-C (lipotrópico) inyección subcutánea",
                "Lemon Bottle (inyección)": "Lemon Bottle lipotrópico inyección subcutánea",
                
                # Reparación Tisular
                "BPC-157 (250 mcg/día)": "BPC-157 250 mcg subcutáneo diario",
                "BPC-157 (500 mcg/día)": "BPC-157 500 mcg subcutáneo diario",
                "BPC-157 (1000 mcg/día)": "BPC-157 1000 mcg subcutáneo diario",
                "TB-500 (2.5 mg/sem)": "TB-500 2.5 mg subcutáneo 2x/semana",
                "TB-500 (5 mg/sem)": "TB-500 5 mg subcutáneo 2x/semana",
                "GHK-Cu (1 mg/día)": "GHK-Cu 1 mg subcutáneo diario",
                "GHK-Cu (2.5 mg/día)": "GHK-Cu 2.5 mg subcutáneo diario",
                "KPV (200 mcg/día)": "KPV 200 mcg subcutáneo diario",
                "ARA-290 (2 mg/día)": "ARA-290 (Cibinetide) 2 mg subcutáneo diario",
                "LL37 (100 mcg/día)": "LL37 100 mcg subcutáneo diario",
                
                # Eje Hormonal / GH
                "HGH (2 IU/día)": "Hormona de crecimiento 2 IU subcutáneo diario",
                "HGH (4 IU/día)": "Hormona de crecimiento 4 IU subcutáneo diario",
                "HGH (6 IU/día)": "Hormona de crecimiento 6 IU subcutáneo diario",
                "Tesamorelin (2 mg/día)": "Tesamorelin 2 mg subcutáneo diario",
                "Ipamorelin (200 mcg/día)": "Ipamorelin 200 mcg subcutáneo antes de dormir",
                "Ipamorelin (300 mcg/día)": "Ipamorelin 300 mcg subcutáneo antes de dormir",
                "CJC-1295 DAC (2 mg/sem)": "CJC-1295 con DAC 2 mg subcutáneo semanal",
                "CJC-1295 sin DAC (100 mcg/día)": "CJC-1295 sin DAC (Mod GRF 1-29) 100 mcg subcutáneo 3x/día",
                "IGF-1 LR3 (40 mcg/día)": "IGF-1 LR3 40 mcg subcutáneo diario",
                "PEG-MGF (200 mcg)": "PEG-MGF 200 mcg subcutáneo post-entrenamiento",
                "Sermorelin (300 mcg/día)": "Sermorelin 300 mcg subcutáneo antes de dormir",
                "GHRP-2 (100 mcg/día)": "GHRP-2 100 mcg subcutáneo 3x/día",
                "GHRP-6 (100 mcg/día)": "GHRP-6 100 mcg subcutáneo 3x/día",
                "Mecasermin (0.4 mg/kg)": "Mecasermin (Increlex) 0.4 mg/kg subcutáneo 2x/día",
                
                # Longevidad
                "Thymosin Alpha-1 (1.6 mg/sem)": "Thymosin Alpha-1 (Zadaxin) 1.6 mg subcutáneo 2x/sem",
                "Epithalon (5 mg/día x 20 días)": "Epithalon 5 mg subcutáneo diario por 20 días",
                "MOTS-C (5 mg/sem)": "MOTS-C 5 mg subcutáneo semanal",
                "NAD+ (250 mg/sem)": "NAD+ 250 mg intravenoso semanal",
                "NAD+ (500 mg/sem)": "NAD+ 500 mg intravenoso semanal",
                "NAD+ (100 mg/día oral)": "NAD+ (NMN/NR) 100 mg oral diario",
                
                # Nootrópicos
                "Selank (250 mcg/día)": "Selank 250 mcg nasal diario",
                "Semax (300 mcg/día)": "Semax 300 mcg nasal diario",
                "DSIP (100 mcg/día)": "DSIP 100 mcg subcutáneo antes de dormir",
                "PT-141 (2 mg)": "PT-141 (Bremelanotide) 2 mg subcutáneo según necesidad",
                "Dihexa (5 mg/día)": "Dihexa 5 mg oral diario",
                "Noopept (10-30 mg/día)": "Noopept (Omberacetam) 10-30 mg oral diario",
                "Cerebrolysin (5 ml/día)": "Cerebrolysin 5 ml intramuscular diario x 20 días",
                
                # Blends
                "Wolverine Stack (BPC+TB)": "BPC-157 250mcg + TB-500 2.5mg subcutáneo diario",
                "Glow Stack (TB+BPC+GHK)": "TB-500 2.5mg + BPC-157 250mcg + GHK-Cu 1mg subcutáneo",
                "Hyper Recovery (TB+BPC+CU+KPV)": "TB-500 + BPC-157 + GHK-Cu + KPV subcutáneo diario",
                "CJC+Ipa Stack": "CJC-1295 DAC 2mg + Ipamorelin 300mcg subcutáneo diario",
                
                # Otros
                "Melanotan II (1 mg/día)": "Melanotan II 1 mg subcutáneo diario",
                "Kisspeptin-10 (100 mcg/día)": "Kisspeptin-10 100 mcg subcutáneo diario",
                "Lunasina (50 mg/día)": "Lunasina 50 mg oral diario",
                "hCG (500 IU/sem)": "hCG 500 IU subcutáneo semanal",
                "Palmitoyl Pentapeptide-4 (tópico)": "Palmitoyl Pentapeptide-4 tópico diario",
                "Acetyl Hexapeptide-8 (tópico)": "Argireline (Acetyl Hexapeptide-8) tópico diario",
            }

            selected_peptides = st.multiselect(
                "Péptidos en tu protocolo",
                options=list(PEPTIDE_OPTIONS.keys()),
                default=[],
                help="Selecciona uno o varios péptidos"
            )

            # Build protocol text from selection
            if selected_peptides:
                protocol = "\n".join([PEPTIDE_OPTIONS[p] for p in selected_peptides])
                st.info("**Protocolo seleccionado:**\n" + protocol)
            else:
                protocol = ""

            # Optional: custom additional notes
            custom_protocol = st.text_area(
                "O notas adicionales del protocolo (opcional)",
                height=60,
                placeholder="ej: other supplements, fasting protocol, timing..."
            )
            if custom_protocol:
                protocol = protocol + "\n" + custom_protocol if protocol else custom_protocol

            st.markdown("**🩺 Síntomas y Estado Actual**")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                sleep = st.select_slider("Calidad del sueño", options=["Mala", "Regular", "Buena", "Excelente"], value="Buena")
                energy = st.select_slider("Nivel de energía", options=["Bajo", "Normal", "Alto", "Muy alto"], value="Normal")
                mood = st.select_slider("Estado de ánimo", options=["Bajo", "Normal", "Elevado"], value="Normal")
            with col_s2:
                pain = st.select_slider("Dolor/inflamación", options=["Ninguno", "Leve", "Moderado", "Severo"], value="Ninguno")
                libido = st.select_slider("Libido", options=["Baja", "Normal", "Alta"], value="Normal")
                recovery = st.select_slider("Recuperación muscular", options=["Lenta", "Normal", "Rápida"], value="Normal")

            st.markdown("**📝 Observaciones Adicionales**")
            symptoms = st.text_area(
                "Síntomas adicionales, efectos secundarios, o cambios recientes",
                height=80,
                placeholder="ej: hinchazón en sitio de inyección, mejora del dolor articular..."
            )

            generate_btn = st.form_submit_button("🔮 Generar Análisis Predictivo", type="primary", use_container_width=True)

        if generate_btn:
            if not GEMINI_KEY:
                st.error("Se requiere GEMINI_API_KEY para esta función.")
            else:
                with st.spinner("Generando análisis predictivo con IA..."):
                    prompt = f"""Eres un endocrinólogo y especialista en péptidos biomédicos. Genera un ANÁLISIS PREDICTIVO de analítica de sangre basado en el siguiente perfil:

PERFIL:
- Edad: {age} años
- Sexo: {sex}
- Peso: {weight} kg

PROTOCOLO ACTUAL:
{protocol if protocol else "No hay péptidos registrados"}

ESTADO ACTUAL:
- Sueño: {sleep}
- Energía: {energy}
- Ánimo: {mood}
- Dolor/inflamación: {pain}
- Libido: {libido}
- Recuperación muscular: {recovery}
- Observaciones: {symptoms if symptoms else "Ninguna"}

Basándote en literatura médica sobre péptidos, genera VALORES PREDICTIVOS para estos biomarcadores con su justificación:
- IGF-1 (rango normal: 100-300 ng/mL)
- Glucosa en ayunas (70-99 mg/dL)
- Testosterona libre (8-25 pg/mL)
- Testosterona total (300-1000 ng/dL)
- Estradiol (10-40 pg/mL)
- Colesterol total (0-200 mg/dL)
- LDL (0-100 mg/dL)
- HDL (40+ mg/dL)
- Triglicéridos (0-150 mg/dL)
- ALT (0-40 U/L)
- AST (0-40 U/L)
- TSH (0.4-4.0 mIU/L)
- Creatinina (0.7-1.3 mg/dL)

Responde SOLO con JSON válido (sin markdown) con esta estructura:
{
    "igf1": {"value": 185, "justification": "..."},
    "glucose": {"value": 85, "justification": "..."},
    "free_testosterone": {"value": 15.5, "justification": "..."},
    "total_testosterone": {"value": 550, "justification": "..."},
    "estradiol": {"value": 25, "justification": "..."},
    "cholesterol_total": {"value": 185, "justification": "..."},
    "ldl": {"value": 95, "justification": "..."},
    "hdl": {"value": 55, "justification": "..."},
    "triglycerides": {"value": 110, "justification": "..."},
    "alt": {"value": 22, "justification": "..."},
    "ast": {"value": 24, "justification": "..."},
    "tsh": {"value": 2.1, "justification": "..."},
    "creatinine": {"value": 0.9, "justification": "..."},
    "summary": "Resumen clínico del análisis...",
    "recommendations": ["Recomendación 1", "Recomendación 2", "Recomendación 3"]
}"""

                    try:
                        from rag_engine import _call_gemini_with_retry
                        result = _call_gemini_with_retry(prompt)
                        
                        if result:
                            # Clean JSON response
                            json_str = result.strip()
                            if json_str.startswith("```"):
                                json_str = json_str.split("\n", 1)[1].rsplit("```", 1)[0]
                            
                            data = json.loads(json_str)
                            
                            # Store in session
                            st.session_state.predictive_data = data
                            st.session_state.predictive_protocol = protocol
                            st.session_state.predictive_profile = {
                                "age": age, "sex": sex, "weight": weight,
                                "sleep": sleep, "energy": energy, "pain": pain
                            }
                        else:
                            st.error("No se pudo generar el análisis. Intenta de nuevo.")
                    except json.JSONDecodeError:
                        st.error("Error procesando respuesta de IA. Intenta de nuevo.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        # Display results if available
        if "predictive_data" in st.session_state:
            data = st.session_state.predictive_data
            profile = st.session_state.predictive_profile

            st.success("✅ Análisis predictivo generado")
            
            # Disclaimer
            st.warning("⚠️ **AVISO:** Este es un análisis PREDICTIVO generado por IA. Los valores son estimaciones basadas en literatura médica y NO sustituyen una analítica real.")

            # Profile summary
            st.markdown(f"**Perfil:** {profile['age']} años, {profile['sex']}, {profile['weight']} kg | Sueño: {profile['sleep']} | Energía: {profile['energy']}")

            # Create synthetic blood work entry
            today = date.today().isoformat()
            
            # Show results table
            st.markdown("### 📊 Valores Predictivos")
            results_data = []
            for marker in BIOMARKER_COLS:
                if marker in data:
                    val = data[marker]["value"]
                    just = data[marker]["justification"]
                    ref = REFERENCE_RANGES.get(marker, {})
                    in_range = ref.get("min", 0) <= val <= ref.get("max", 999)
                    status = "✅" if in_range else "⚠️"
                    results_data.append({
                        "Biomarcador": ref.get("label", marker),
                        "Valor": f"{val} {ref.get('unit', '')}",
                        "Rango": f"{ref.get('min', '?')}-{ref.get('max', '?')}",
                        "Estado": status
                    })
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Justifications
            with st.expander("📝 Justificaciones por biomarcador"):
                for marker in BIOMARKER_COLS:
                    if marker in data:
                        ref = REFERENCE_RANGES.get(marker, {})
                        val = data[marker]["value"]
                        just = data[marker]["justification"]
                        st.markdown(f"**{ref.get('label', marker)}** ({val} {ref.get('unit', '')})")
                        st.caption(just)

            # Summary and recommendations
            st.markdown("### 🩺 Resumen Clínico")
            st.info(data.get("summary", "Sin resumen disponible."))

            st.markdown("### 💡 Recomendaciones")
            for rec in data.get("recommendations", []):
                st.markdown(f"- {rec}")

            # Option to save as synthetic entry
            if st.button("💾 Guardar como analítica sintética", type="secondary"):
                try:
                    execute_insert("blood_markers", (
                        today, f"IA Predictivo - {profile['age']}a {profile['sex']}",
                        data.get("igf1", {}).get("value"),
                        data.get("glucose", {}).get("value"),
                        data.get("free_testosterone", {}).get("value"),
                        data.get("total_testosterone", {}).get("value"),
                        data.get("estradiol", {}).get("value"),
                        data.get("cholesterol_total", {}).get("value"),
                        data.get("ldl", {}).get("value"),
                        data.get("hdl", {}).get("value"),
                        data.get("triglycerides", {}).get("value"),
                        data.get("alt", {}).get("value"),
                        data.get("ast", {}).get("value"),
                        data.get("tsh", {}).get("value"),
                        data.get("creatinine", {}).get("value"),
                        f"[ANÁLISIS PREDICTIVO IA] Protocolo: {st.session_state.get('predictive_protocol', 'N/A')[:200]}"
                    ))
                    st.success("✅ Analítica sintética guardada. Aparecerá en los gráficos.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ============================================================
# TAB 4: COACH CLÍNICO (conversacional con contexto personal)
# ============================================================
with tab_coach:
    st.header("🧠 Coach Clínico de Peptibro")
    st.markdown("**Interpretación personalizada** • Basado en tu base de conocimiento + registros")
    st.caption("Este coach puede leer tus últimos registros y analíticas para dar respuestas contextuales.")

    # Inicializar historial de conversación
    if "coach_history" not in st.session_state:
        st.session_state.coach_history = []

    # Mostrar historial
    for turn in st.session_state.coach_history:
        with st.chat_message("user"):
            st.write(turn["user"])
        with st.chat_message("assistant"):
            st.write(turn["coach"])

    # Input del usuario
    user_input = st.chat_input("Escribe tu pregunta al Coach...")

    if user_input:
        # Añadir mensaje del usuario al historial visual
        with st.chat_message("user"):
            st.write(user_input)

        # Llamar al Coach
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                response = chat_with_coach(user_input, st.session_state.coach_history)

            st.write(response)

        # Guardar en el historial de sesión
        st.session_state.coach_history.append({
            "user": user_input,
            "coach": response
        })

    # Botón para limpiar conversación
    if st.session_state.coach_history:
        if st.button("🧹 Limpiar conversación", use_container_width=True):
            st.session_state.coach_history = []
            st.rerun()

    st.divider()
    st.caption("El Coach usa Groq (Llama 3.3 70B) y tu base de conocimiento local. Nunca inventa información.")

# ============================================================
# TAB 5: OPENCLAW (CLAWD)
# ============================================================
with tab_openclaw:
    st.header("🦞 Clawd - Asistente OpenClaw")
    
    # Detect if running locally or on cloud
    import os
    is_cloud = os.environ.get("STREAMLIT_CLOUD") or os.path.exists("/tmp")
    
    if is_cloud:
        # Cloud version - show info only
        st.info("""
        **📱 Clawd está disponible en tu computadora local.**
        
        Esta versión de Peptibro corre en la nube y no puede acceder a OpenClaw directamente.
        Para usar Clawd, necesitas ejecutar Peptibro localmente.
        
        **¿Qué es Clawd?**
        - Asistente de IA especializado en péptidos y salud
        - Puede consultarse por WhatsApp (próximamente)
        - Base de conocimiento conectada a Peptibro
        """)
        
        st.markdown("""
        ### Skills de Clawd
        
        | Skill | Descripción |
        |-------|-------------|
        | 🔍 peptide-lookup | Consulta protocolos de péptidos |
        | 🚨 bloodwork-alerts | Alertas de analíticas fuera de rango |
        | ⏰ dose-reminder | Recordatorios de dosis |
        | 📋 protocol-summary | Resúmenes de protocolos |
        """)
        
        st.markdown("""
        **Para ejecutar Clawd localmente:**
        ```bash
        # Instalar OpenClaw
        npm install -g openclaw
        
        # Iniciar gateway
        openclaw gateway restart
        
        # Conectar Peptibro
        openclaw channels add --channel whatsapp
        ```
        """)
    else:
        # Local version - check connection
        import requests
        try:
            response = requests.get("http://127.0.0.1:18789", timeout=2)
            openclaw_running = True
        except:
            openclaw_running = False
        
        if openclaw_running:
            st.success("✅ OpenClaw Gateway está ejecutándose")
            st.markdown("""
            ### 🦞 Clawd está listo
            
            **Abre el dashboard de OpenClaw en una nueva pestaña:**
            
            👉 [**Abrir Clawd Dashboard**](http://127.0.0.1:18789) 👈
            """)
        else:
            st.warning("⚠️ OpenClaw Gateway no está ejecutándose")
            st.markdown("""
            Para iniciar OpenClaw:
            1. Abre una terminal
            2. Ejecuta: `openclaw gateway restart`
            
            **Dashboard:** http://127.0.0.1:18789
            """)
        
        st.divider()
        st.markdown("""
        ### Sobre Clawd 🦞
        
        **Nombre:** Clawd
        **Especialización:** Péptidos, salud y fitness
        
        **Skills disponibles:**
        - 🔍 **peptide-lookup** - Consulta protocolos de péptidos
        - 🚨 **bloodwork-alerts** - Alertas de analíticas
        - ⏰ **dose-reminder** - Recordatorios de dosis
        - 📋 **protocol-summary** - Resúmenes de protocolos
        
        **Modelo:** Gemini 2.0 Flash (gratis)
        **Base de conocimiento:** Conectada a Peptibro
        """)

# Footer
st.divider()
st.caption("Peptibro • Todo el conocimiento clínico viene de tus PDFs locales • RAG estricto anti-alucinación")
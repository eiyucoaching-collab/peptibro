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
from rag_engine import query_peptide_protocol, ingest_knowledge_base, has_knowledge_base, chat_with_coach

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
tab_log, tab_oraculo, tab_dashboard, tab_coach, tab_openclaw = st.tabs([
    "📝 Log Diario",
    "📖 Biblioteca Clínica (El Oráculo)",
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
# TAB 2: BIBLIOTECA CLÍNICA (ORÁCULO RAG)
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
        conn = get_connection()
        df_protocol = pd.read_sql("""
            SELECT DISTINCT compound_name, AVG(dosage_mcg) as avg_dose, COUNT(*) as days
            FROM daily_log 
            WHERE date >= date('now', '-90 days')
            GROUP BY compound_name
        """, conn)
        conn.close()

        default_protocol = ""
        if not df_protocol.empty:
            protocol_lines = []
            for _, row in df_protocol.iterrows():
                protocol_lines.append(f"- {row['compound_name']}: {int(row['avg_dose'])} mcg x {int(row['days'])} días")
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
            protocol = st.text_area(
                "Péptidos y dosis",
                value=default_protocol,
                height=120,
                placeholder="ej: BPC-157 250mcg, CJC-1295 2000mcg..."
            )

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
    
    # Verificar si el servidor de OpenClaw está corriendo
    import requests
    try:
        response = requests.get("http://127.0.0.1:18789", timeout=2)
        openclaw_running = True
    except:
        openclaw_running = False
    
    if openclaw_running:
        st.success("✅ OpenClaw Gateway está ejecutándose")
        
        # Direct link instead of iframe (Streamlit blocks cross-origin iframes)
        st.markdown("""
        ### 🦞 Clawd está listo
        
        **Abre el dashboard de OpenClaw en una nueva pestaña:**
        
        👉 [**Abrir Clawd Dashboard**](http://127.0.0.1:18789) 👈
        
        *Haz clic en el enlace de arriba para abrir el asistente Clawd.*
        """)
    else:
        st.warning("⚠️ OpenClaw Gateway no está ejecutándose")
        st.markdown("""
        Para iniciar OpenClaw:
        1. Abre una terminal
        2. Ejecuta: `openclaw gateway restart`
        3. O ejecuta: `C:\\Users\\FUJITSU\\Peptibro\\start_peptibro.bat`
        
        **Dashboard:** http://127.0.0.1:18789
        """)
    
    # Información del asistente
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
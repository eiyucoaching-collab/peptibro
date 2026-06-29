"""
Peptibro - Motor RAG Clinico (Cloud-ready)
Uses LangChain + ChromaDB + HuggingFace Embeddings
Supports both local and Streamlit Community Cloud deployment.
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from groq import Groq

# === SECRETS MANAGEMENT ===
def _get_secret(key: str) -> str:
    """Get secret from Streamlit secrets or environment variable."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, "")

# === EMBEDDINGS ===
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    USE_LOCAL_EMBEDDINGS = True
except ImportError:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    USE_LOCAL_EMBEDDINGS = False

# === PATHS ===
BASE_KNOWLEDGE_DIR = Path("Base_Conocimiento")
COLLECTION_NAME = "peptibro_clinical"

def _get_chroma_dir() -> Path:
    """Get ChromaDB directory based on environment."""
    import sys
    if sys.platform == "linux":  # Streamlit Cloud
        return Path("/tmp") / "chroma_db"
    return Path("db") / "chroma_langchain_db"

CHROMA_PERSIST_DIR = _get_chroma_dir()
VERSION_FILE = CHROMA_PERSIST_DIR.parent / "knowledge_version.json"

# === CREDENTIALS ===
GEMINI_KEY = _get_secret("GEMINI_API_KEY")
if GEMINI_KEY and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = GEMINI_KEY

GROQ_KEY = _get_secret("GROQ_API_KEY")

# === INITIALIZE COMPONENTS ===
@st.cache_resource
def _init_embeddings():
    """Initialize embeddings (cached)."""
    if USE_LOCAL_EMBEDDINGS:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = _init_embeddings()

@st.cache_resource
def _init_vectorstore():
    """Initialize ChromaDB vectorstore (cached)."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

vectorstore = _init_vectorstore()

@st.cache_resource
def _init_groq():
    """Initialize Groq client (cached)."""
    if GROQ_KEY:
        return Groq(api_key=GROQ_KEY)
    return None

groq_client = _init_groq()


# === HELPER FUNCTIONS ===
def format_docs_with_sources(docs: list) -> str:
    """Format documents with source citations."""
    formatted_chunks = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page", "?")
        chunk_text = doc.page_content.strip()
        formatted_chunks.append(f"[Fuente: {source} | Pagina: {page}]\n{chunk_text}")
    return "\n\n---\n\n".join(formatted_chunks)


def _file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file."""
    return hashlib.md5(filepath.read_bytes()).hexdigest()


def _load_version_tracker() -> dict:
    """Load version tracker from disk."""
    if VERSION_FILE.exists():
        try:
            return json.loads(VERSION_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_version_tracker(data: dict):
    """Save version tracker to disk."""
    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    VERSION_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_changed_files() -> list:
    """Detect which files have changed since last ingestion."""
    tracker = _load_version_tracker()
    base_path = Path(BASE_KNOWLEDGE_DIR)
    changed = []

    for filepath in base_path.rglob("*"):
        if filepath.is_file() and filepath.suffix.lower() in {".pdf", ".md", ".markdown", ".txt"}:
            rel = str(filepath.relative_to(base_path))
            current_hash = _file_hash(filepath)
            if tracker.get(rel) != current_hash:
                changed.append(filepath)
                tracker[rel] = current_hash

    _save_version_tracker(tracker)
    return changed


# === KNOWLEDGE INGESTION ===
def ingest_knowledge_base(force_rebuild: bool = False, incremental: bool = True):
    """Ingest PDFs and Markdown from Base_Conocimiento."""
    global vectorstore

    if force_rebuild:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        vectorstore = _init_vectorstore()
        _save_version_tracker({})

    # Determine files to process
    if incremental and not force_rebuild:
        files_to_process = _get_changed_files()
        if not files_to_process:
            print("Ingesta incremental: no hay archivos nuevos.")
            return
    else:
        base_path = Path(BASE_KNOWLEDGE_DIR)
        files_to_process = [
            f for f in base_path.rglob("*")
            if f.is_file() and f.suffix.lower() in {".pdf", ".md", ".markdown", ".txt"}
        ]

    # Load documents
    documents = []
    for filepath in files_to_process:
        try:
            if filepath.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(filepath))
            else:
                loader = TextLoader(str(filepath), encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = str(filepath)
            documents.extend(docs)
        except Exception as e:
            print(f"Error cargando {filepath.name}: {e}")

    if not documents:
        print("No se encontraron archivos para ingerir.")
        return

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        src_path = chunk.metadata.get("source", "unknown")
        chunk.metadata["source"] = Path(src_path).name
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = chunk.metadata.get("page_number", "N/A")

    vs = _init_vectorstore()
    vs.add_documents(chunks)
    vectorstore = vs

    # Save version
    version_data = {
        "last_ingestion": datetime.now().isoformat(),
        "total_chunks": vs._collection.count(),
        "files_processed": len(files_to_process)
    }
    _save_version_tracker(_load_version_tracker() | {"_version": version_data})
    print(f"Ingesta completada: {len(chunks)} chunks de {len(documents)} archivos.")


# === KNOWLEDGE BASE STATUS ===
def has_knowledge_base() -> bool:
    """Check if knowledge base has data."""
    try:
        count = vectorstore._collection.count()
        return count > 0
    except Exception:
        return False


# === ORACLE (RAG QUERY) ===
def _call_gemini_with_retry(prompt: str, max_retries: int = 3):
    """Call Gemini with automatic retries."""
    if not GEMINI_KEY:
        return None
    models = ["gemini-2.0-flash", "gemini-1.5-flash"]
    import time
    for model_name in models:
        for attempt in range(max_retries):
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
                chain = ChatPromptTemplate.from_template("{input}") | llm | StrOutputParser()
                return chain.invoke({"input": prompt})
            except Exception:
                if attempt == max_retries - 1:
                    continue
                time.sleep(2 ** attempt)
    return None


def query_peptide_protocol(compound_name: str) -> str:
    """Query the RAG oracle for peptide information."""
    if not compound_name or not compound_name.strip():
        return "Especifica un nombre de peptido/composto."

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    docs = retriever.invoke(compound_name.strip())
    context = format_docs_with_sources(docs)

    if not docs:
        return "No hay datos clinicos locales sobre esto."

    system_prompt = f"""Eres un asistente de investigacion clinica extremadamente preciso.

INSTRUCCIONES ESTRICTAS:
- Usa UNICAMENTE el texto medico recuperado en el contexto.
- Si la dosis, ciclo, frecuencia, sinergia o protocolo del compuesto NO aparece explicitamente en el contexto, responde EXACTAMENTE:
  "No hay datos clinicos locales sobre esto."
- NUNCA inventes dosis.
- Cuando proporciones informacion, cita SIEMPRE la fuente al final.

PREGUNTA: {compound_name}

CONTEXTO:
{context}

RESPUESTA:"""

    response = _call_gemini_with_retry(system_prompt)
    if response:
        return response.strip()
    return (
        "**Gemini no disponible en este momento.**\n\n"
        "Mostrando informacion directamente de tu base de conocimiento local:\n\n"
        + context
    )


# === COACH CLINICAL ===
def get_user_context() -> str:
    """Get user context from database."""
    from database_setup import get_connection
    import pandas as pd

    conn = get_connection()
    df_log = pd.read_sql("""
        SELECT date, compound_name, dosage_mcg, notes
        FROM daily_log ORDER BY date DESC LIMIT 30
    """, conn)

    df_blood = pd.read_sql("""
        SELECT date, test_name, igf1, glucose, free_testosterone, total_testosterone,
               estradiol, alt, ast, tsh
        FROM blood_markers ORDER BY date DESC LIMIT 5
    """, conn)
    conn.close()

    context = "=== ULTIMOS REGISTROS DEL USUARIO ===\n"
    if not df_log.empty:
        context += "\n**Log Diario (ultimos 30 registros):**\n"
        for _, row in df_log.iterrows():
            context += f"- {row['date']}: {row['compound_name']} {row['dosage_mcg']} mcg"
            if row['notes']:
                context += f" | Notas: {row['notes']}"
            context += "\n"
    else:
        context += "\n**Log Diario:** Sin registros aun.\n"

    if not df_blood.empty:
        context += "\n**Ultimas Analiticas (5 mas recientes):**\n"
        for _, row in df_blood.iterrows():
            context += f"- {row['date']} ({row['test_name']}):\n"
            for col in ["igf1", "glucose", "free_testosterone", "total_testosterone", "estradiol", "alt", "ast", "tsh"]:
                val = row[col]
                if pd.notna(val):
                    context += f"  {col.upper()}: {val}\n"
    else:
        context += "\n**Analiticas:** Sin registros aun.\n"

    return context


def chat_with_coach(user_message: str, conversation_history: list = None) -> str:
    """Chat with the Clinical Coach using Groq."""
    if not groq_client:
        return "Error: GROQ_API_KEY no configurada en secrets."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(user_message)
    knowledge_context = "\n\n".join([doc.page_content for doc in docs])
    user_context = get_user_context()

    history_text = ""
    if conversation_history:
        history_text = "\n".join([
            f"Usuario: {turn['user']}\nCoach: {turn['coach']}"
            for turn in conversation_history[-6:]
        ])

    system_prompt = f"""Eres el Coach Clinico de Peptibro.

Tu rol es ayudar al usuario a entender y gestionar sus protocolos de peptidos de forma segura y basada en evidencia local.

REGLAS OBLIGATORIAS:
1. Solo puedes usar informacion que exista en la base de conocimiento local o en los registros personales del usuario.
2. Si no tienes datos suficientes, di exactamente: "No hay datos clinicos locales suficientes para responder esta pregunta con certeza."
3. Nunca inventes dosis, protocolos ni combinaciones.
4. Puedes explicar, interpretar resultados de analiticas, sugerir que biomarcadores monitorizar y analizar patrones.
5. Cuando sugieras algo, siempre explica el razonamiento basado en la informacion disponible.
6. Sé claro, directo y profesional. Usa español.

Contexto disponible:
- Base de conocimiento de peptidos (recuperada por RAG)
- Ultimos registros del usuario (daily_log)
- Ultimas analiticas del usuario (blood_markers)"""

    user_prompt = f"""
=== CONTEXTO DE CONOCIMIENTO (ChromaDB) ===
{knowledge_context}

=== CONTEXTO PERSONAL DEL USUARIO ===
{user_context}

=== HISTORIAL DE CONVERSACION ===
{history_text}

=== PREGUNTA ACTUAL DEL USUARIO ===
{user_message}

Responde como el Coach Clinico de Peptibro."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al consultar el Coach: {str(e)}"

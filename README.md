# Peptibro

**Seguimiento de Protocolos de Peptidos con IA Clinica**

App web para rastrear dosis de peptidos, analiticas de sangre, y consultar una base de conocimiento clinica local usando RAG (Retrieval-Augmented Generation).

## Caracteristicas

- **Log Diario** - Registra tus dosis de peptidos
- **Oraculo Clinico** - Consulta protocolos desde tu base de conocimiento local (RAG anti-alucinacion)
- **Coach Clinico** - Chat conversacional con contexto personal
- **Dashboard de Analiticas** - Graficos de evolucion de biomarcadores
- **Exportacion PDF** - Informes mensuales

## Deploy Gratis

### Streamlit Community Cloud

1. Haz fork de este repo
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub
4. Selecciona este repo y el archivo `app.py`
5. En Secrets, agrega tus API keys:

```toml
GEMINI_API_KEY = "tu_api_key"
GROQ_API_KEY = "tu_api_key"
```

6. Click Deploy

### Ejecutar Localmente

```bash
# Instalar dependencias
pip install -r requirements.txt

# Crear archivo .env
cp .env.example .env
# Edita .env con tus API keys

# Ejecutar
streamlit run app.py
```

## API Keys Necesarias

- **Gemini API** - Para el Oraculo Clinico (gratis)
- **Groq API** - Para el Coach Clinico (gratis)

## Tecnologias

- Streamlit - Frontend
- LangChain - Framework RAG
- ChromaDB - Vector database
- HuggingFace - Embeddings locales
- Gemini - LLM para Oraculo
- Groq - LLM para Coach
- SQLite - Base de datos local

## License

MIT

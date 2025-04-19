# 🧠 EDT LangChain AI Agent

This project is a LangChain + LangGraph-based conversational AI system for a medical office assistant. It reads PDF documents, creates a vector database using ChromaDB, and allows natural language interaction via API powered by OpenAI models.

---

## 🚀 Features

- ✅ Conversational agent powered by LangGraph and LangChain
- ✅ OpenAI integration for structured output and tool calling
- ✅ PDF ingestion and vector store creation (RAG-based)
- ✅ API endpoint using FastAPI for external queries
- ✅ Visual representation of the state machine using Mermaid

---

## 🧱 Project Structure

```
edt-langchain-ai-agent/
├── data/                   # Data files (e.g., PDFs)
├── docs/                   # (Optional) Documentation assets
├── src/                    # Core application logic
│   ├── __init__.py
│   ├── config.py           # Loads settings from environment
│   ├── paths.py            # File paths (PDF, DB, diagram)
│   ├── chromadb_manager.py # Wrapper around Chroma vector store
│   ├── create_embeddings.py# Script to load PDF and index embeddings
│   └── main.py             # Main app: LangGraph + FastAPI
├── .env                    # Environment variables (not versioned)
├── .env.example            # Example env file for setup
├── environment.yml         # Conda environment definition
├── pyproject.toml          # Project metadata for pip editable install
├── requirements.txt        # PIP dependencies
├── LICENSE
└── README.md               # You're here!
```

---

## 🛠️ Setup

### 1. Clone and configure environment

```bash
git clone git@github.com:jasonssdev/edt-langchain-ai-agent.git
cd edt-langchain-ai-agent
conda env create -f environment.yml
conda activate ai-py3.12
pip install -e .
```

### 2. Add environment variables

Copy the `.env.example` file and fill in your OpenAI API key:

```bash
cp .env.example .env
```

---

## 📄 Usage

### Create embeddings from a PDF

```bash
python -m src.create_embeddings
```

### Run the API with FastAPI

```bash
uvicorn src.main:app --reload --port 8000
```

### Make a request

POST to `http://localhost:8000/run`

```json
{
  "question": "What is the price of a pediatric consultation?"
}
```

---

## 📊 Architecture

- **LangGraph** manages the conversational state machine
- **LangChain** handles LLMs, structured output, tools
- **Chroma** stores vector embeddings from PDF
- **OpenAI** provides GPT model for querying and responses

---

## 👨‍💼 Author

Made with ❤️ by [@jasonssdev](https://github.com/jasonssdev)

---

## 📄 License

[Apache 2.0](./LICENSE)


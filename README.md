# RAG-Powered Document Action Extractor

A Retrieval-Augmented Generation (RAG) application that automatically extracts action items, owners, deadlines, and priority levels from unstructured documents (PDF, DOCX, TXT) using LangChain, ChromaDB, and Google Gemini.

---

## 🚀 Features

- **Multi-Format Ingestion**: Supports `.pdf` (via `pypdf`), `.docx` (via `python-docx`), and plain `.txt` files.
- **Smart Chunking**: Recursively chunks text using LangChain's `RecursiveCharacterTextSplitter`.
- **Vector Search**: Embeds and stores chunks in a persistent local ChromaDB instance with Google Generative AI embeddings.
- **Structured Action Extraction**: Uses Gemini to extract structured task data (task, owner, deadline, priority, and source quotation) validated with Pydantic.
- **Interactive Web UI**: Clean Streamlit interface for uploading files, triggering ingestion, and querying action items.

---

## 🛠️ Tech Stack

- **Python** (3.10+)
- **Streamlit**: Web interface
- **LangChain & LangChain Community**: Orchestration, chunking, and LLM integrations
- **ChromaDB**: Local vector database
- **Google Gemini**: Embeddings and chat language models
- **Pydantic**: Structured output schema and validation

---

## 📦 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Darsh505/rag-action-extractor.git
cd rag-action-extractor
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Copy the `.env.example` file to `.env` and add your Google Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GOOGLE_API_KEY=your_actual_api_key_here
```

### 5. Run the application
```bash
streamlit run app.py
```
The app will be accessible at `http://localhost:8501`.

---

## 📂 Project Structure

```text
rag-action-extractor/
├── app.py              # Streamlit web application
├── ingest.py           # Document loading, text chunking & vector storage
├── rag.py              # ChromaDB vector retrieval & LLM action extraction
├── models.py           # Pydantic schemas (ActionItem, ActionItemList)
├── utils.py            # Utility functions
├── requirements.txt    # Project dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git exclusion rules
├── data/               # Sample document storage
└── README.md           # Project documentation
```

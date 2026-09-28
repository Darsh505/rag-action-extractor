import streamlit as st

st.set_page_config(
    page_title="RAG-Powered Document Action Extractor",
    page_icon="📋",
    layout="wide",
)

# App Header
st.title("📋 RAG-Powered Document Action Extractor")
st.markdown(
    """
    Transform unstructured documents into structured, actionable intelligence.
    Upload meeting minutes, reports, or project specs to chunk and index them into a local
    vector store, then extract assigned tasks, owners, deadlines, and priorities using Gemini RAG.
    """
)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Supported Formats")
    st.markdown(
        """
        You can upload the following document types:
        - **PDF (`.pdf`)**: Extracted via `pypdf`
        - **Word Document (`.docx`)**: Extracted via `python-docx`
        - **Plain Text (`.txt`)**: Read directly as UTF-8 text

        Ensure your document contains notes, minutes, or tasks for optimal extraction.
        """
    )
    st.divider()
    st.caption("⚡ Built with LangChain, ChromaDB & Google Gemini")

# Section 1: Upload & Ingest
st.header("1. Upload & Ingest")
st.markdown("Upload a document to parse its contents and store vectorized embeddings in ChromaDB.")
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt"],
    help="Select a PDF, DOCX, or TXT document to process and index.",
)

# Section 2: Extract Actions
st.divider()
st.header("2. Extract Actions")
st.markdown("Query the ingested document store to identify, structure, and tabularize action items.")
query_input = st.text_input(
    "Query / Instruction",
    value="Extract all action items",
    placeholder="e.g., Extract all action items, owners, and deadlines",
)
st.button("🚀 Extract Action Items", type="primary")

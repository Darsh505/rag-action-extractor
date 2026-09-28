import os
import streamlit as st
from ingest import load_document, chunk_text, reset_collection, store_chunks

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

process_button = st.button("📥 Process Document", type="primary")

if process_button:
    if uploaded_file is None:
        st.warning("Please select and upload a document first before processing.")
    else:
        try:
            progress_bar = st.progress(0, text="Preparing file...")

            # 1. Save uploaded file to data/uploaded_file.<ext>
            _, ext = os.path.splitext(uploaded_file.name)
            os.makedirs("data", exist_ok=True)
            saved_path = f"data/uploaded_file{ext.lower()}"

            progress_bar.progress(20, text=f"Saving uploaded file to {saved_path}...")
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. Call load_document -> chunk_text -> reset_collection -> store_chunks
            progress_bar.progress(40, text="Extracting text from document...")
            raw_text = load_document(saved_path)

            progress_bar.progress(60, text="Chunking text with LangChain...")
            chunks = chunk_text(raw_text)

            progress_bar.progress(80, text="Resetting ChromaDB collection...")
            reset_collection("documents")

            progress_bar.progress(90, text="Generating embeddings and storing chunks...")
            store_chunks(chunks, collection_name="documents")

            # 3. Show success and complete progress
            progress_bar.progress(100, text="Document processing complete!")
            st.success(f"Ingested {len(chunks)} chunks.")
            st.session_state["ingested_file"] = uploaded_file.name
            st.session_state["chunk_count"] = len(chunks)

        except Exception as e:
            st.error(f"Failed to process document: {e}")

# Section 2: Extract Actions
st.divider()
st.header("2. Extract Actions")
st.markdown("Query the ingested document store to identify, structure, and tabularize action items.")
query_input = st.text_input(
    "Query / Instruction",
    value="Extract all action items",
    placeholder="e.g., Extract all action items, owners, and deadlines",
)
st.button("🚀 Extract Action Items")

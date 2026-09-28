import os
import uuid
from dotenv import load_dotenv
from pypdf import PdfReader
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


class GoogleGenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function adapter for ChromaDB using LangChain's GoogleGenerativeAIEmbeddings."""

    def __init__(self, model: str = "models/embedding-001"):
        # Map deprecated models/embedding-001 to models/gemini-embedding-001 to prevent 404 and rate limit exhaustion
        self.model = "models/gemini-embedding-001" if model == "models/embedding-001" else model
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.model)

    def __call__(self, input: Documents) -> Embeddings:
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.embeddings.embed_documents(input)
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise e




def load_document(file_path: str) -> str:
    """
    Loads and extracts text from a document based on its file extension.

    Supported formats:
    - PDF (.pdf) via pypdf
    - DOCX (.docx) via python-docx
    - Text (.txt) via plain text reader

    Args:
        file_path (str): Path to the target document.

    Returns:
        str: The full extracted text content.

    Raises:
        FileNotFoundError: If the file does not exist at file_path.
        ValueError: If the file extension is unsupported.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        return "\n".join(pages_text)

    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported file types are: .pdf, .docx, .txt"
        )


def chunk_text(text: str) -> list[str]:
    """
    Splits text into chunks using RecursiveCharacterTextSplitter.

    Args:
        text (str): The input text to split.

    Returns:
        list[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    return text_splitter.split_text(text)


def reset_collection(collection_name: str = "documents", db_path: str = "./chroma_db") -> None:
    """
    Clears old data by deleting the specified ChromaDB collection before re-ingesting.

    Args:
        collection_name (str): The name of the collection to delete. Defaults to "documents".
        db_path (str): The path to the ChromaDB directory. Defaults to "./chroma_db".
    """
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        # Collection does not exist or has already been cleared
        pass


def store_chunks(chunks: list[str], collection_name: str = "documents"):
    """
    Creates a ChromaDB persistent client at "./chroma_db", uses
    GoogleGenerativeAIEmbeddings (model: "models/embedding-001"),
    adds chunks to a collection with unique IDs, and returns the collection object.

    Args:
        chunks (list[str]): List of text chunks to embed and store.
        collection_name (str): The name of the collection. Defaults to "documents".

    Returns:
        chromadb.Collection: The ChromaDB collection object.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_function = GoogleGenAIEmbeddingFunction(model="models/embedding-001")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    if chunks:
        ids = [f"{collection_name}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            ids=ids,
        )

    return collection


if __name__ == "__main__":
    sample_file = "data/sample.pdf"

    # 1. Loads "data/sample.pdf"
    text = load_document(sample_file)

    # 2. Chunks it
    chunks = chunk_text(text)

    # 3. Resets the collection
    reset_collection("documents")

    # 4. Stores the chunks
    store_chunks(chunks, collection_name="documents")

    # 5. Prints "Ingestion complete. X chunks stored."
    print(f"Ingestion complete. {len(chunks)} chunks stored.")


import os
import re
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from ingest import GoogleGenAIEmbeddingFunction
from models import ActionItem, ActionItemList

load_dotenv()


def get_llm(model: str = "gemini-1.5-flash", temperature: float = 0) -> ChatGoogleGenerativeAI:
    """
    Returns a ChatGoogleGenerativeAI instance using the specified model and temperature,
    reading the API key from .env via python-dotenv.

    Args:
        model (str): Gemini model name. Defaults to "gemini-1.5-flash".
        temperature (float): Model temperature. Defaults to 0.

    Returns:
        ChatGoogleGenerativeAI: The initialized LLM instance.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )


def retrieve_chunks(
    query: str,
    k: int = 5,
    collection_name: str = "documents",
    db_path: str = "./chroma_db",
) -> list[str]:
    """
    Connects to the existing ChromaDB collection at "./chroma_db",
    embeds the query using GoogleGenerativeAIEmbeddings, and retrieves
    the top-k most relevant chunk texts.

    Args:
        query (str): The search query.
        k (int): Number of most relevant chunks to return. Defaults to 5.
        collection_name (str): The name of the collection. Defaults to "documents".
        db_path (str): The directory of the ChromaDB persistent client. Defaults to "./chroma_db".

    Returns:
        list[str]: The top-k relevant text chunks.
    """
    client = chromadb.PersistentClient(path=db_path)
    embedding_function = GoogleGenAIEmbeddingFunction(model="models/embedding-001")

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
    except Exception:
        return []

    total_docs = collection.count()
    if total_docs == 0:
        return []

    n_results = min(k, total_docs)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    documents = results.get("documents", [])
    if documents and len(documents) > 0:
        return documents[0]
    return []


def extract_actions(query: str) -> list[ActionItem]:
    """
    Extracts action items from document context relevant to the given query.

    1. Retrieves top 5 chunks with retrieve_chunks(query).
    2. Joins them into one context string.
    3. Sends extraction prompt to the LLM.
    4. Parses the JSON response with ActionItemList.model_validate_json().
    5. Returns the list of ActionItem objects.

    Args:
        query (str): Query to search for relevant context.

    Returns:
        list[ActionItem]: Extracted list of ActionItem objects.
    """
    # 1. Retrieves top 5 chunks with retrieve_chunks(query)
    chunks = retrieve_chunks(query, k=5)
    if not chunks:
        return []

    # 2. Joins them into one context string
    context = "\n\n".join(chunks)

    # 3. Sends prompt to LLM
    prompt = f"""You are an action item extractor. Given the context below, extract all action items.
Return ONLY a valid JSON object matching this schema:
{{"action_items": [{{"task": str, "owner": str or null, "deadline": str or null, "priority": str or null, "source": str}}]}}

Rules:
- Do not hallucinate. If a field is unknown, use null.
- 'source' must be a short quote from the context.
- If no action items exist, return {{"action_items": []}}.

Context:
{context}"""

    llm = get_llm(model="gemini-1.5-flash")
    try:
        response = llm.invoke(prompt)



        content = response.content
        if isinstance(content, list):
            text_content = "".join(
                [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
            )
        else:
            text_content = str(content)

        cleaned_content = text_content.strip()
        if cleaned_content.startswith("```"):
            cleaned_content = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned_content)
            cleaned_content = re.sub(r"\n?```$", "", cleaned_content).strip()

        # 4. Parses the JSON response with ActionItemList.model_validate_json()
        parsed = ActionItemList.model_validate_json(cleaned_content)

        # 5. Returns the list of ActionItem objects
        return parsed.action_items
    except Exception as e:
        print(f"Error parsing action items: {e}")
        return []


if __name__ == "__main__":
    query = "Extract all action items"
    print(f"Extracting actions for query: '{query}'\n")

    items = extract_actions(query)
    print(f"Found {len(items)} action item(s):\n")
    for idx, item in enumerate(items, 1):
        owner_str = f"Owner: {item.owner}" if item.owner else "Owner: None"
        deadline_str = f" | Deadline: {item.deadline}" if item.deadline else " | Deadline: None"
        priority_str = f" | Priority: {item.priority}" if item.priority else " | Priority: None"
        print(f"{idx}. Task: {item.task} ({owner_str}{deadline_str}{priority_str})")
        print(f"   Source: \"{item.source}\"\n")


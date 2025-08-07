import streamlit as st
import lancedb
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from logger import get_logger

# Load env (OPENAI_API_KEY, etc.)
load_dotenv()
client = OpenAI()

# Initialize logger
logger = get_logger("streamlit_app")

@st.cache_resource
def init_db():
    """Connect (and cache) the LanceDB table."""
    try:
        db_path = "/Users/discovery/Desktop/Docling/data/lancedb"
        logger.info(f"Attempting to connect to LanceDB at: {db_path}")
        db = lancedb.connect(db_path)
        table = db.open_table("lance_neur_papers_db")
        logger.info("Successfully connected to LanceDB table")
        return table
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        st.error(f"DB connection failed: {e}")
        return None

def search_docs(query: str, table, num_results: int = 5) -> pd.DataFrame:
    """
    Vector-search the table using OpenAI embeddings for the query.
    The table already contains pre-computed embeddings, so we generate 
    an embedding for the query and search against the stored vectors.
    """
    try:
        logger.info(f"Searching for query: '{query}' with limit: {num_results}")
        
        # Generate embedding for the query using OpenAI
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding
        logger.info(f"Generated embedding for query with dimension: {len(query_embedding)}")
        
        # Search using the query embedding directly (LanceDB format)
        df = (
            table
            .search(query_embedding)  # Pass embedding directly, not as query param
            .select(["text", "metadata", "_distance"])  # Select specific columns
            .limit(num_results)
            .to_pandas()
        )
        
        logger.info(f"Search returned {len(df)} results")
        return df
        
    except Exception as e:
        logger.error(f"Search failed for query '{query}': {e}")
        st.error(f"Search failed: {e}")
        return pd.DataFrame()

def build_context_string(df: pd.DataFrame) -> str:
    """Build a single context string for the LLM from the search results."""
    if df.empty:
        logger.warning("No search results to build context from")
        return "No relevant context found."
    
    logger.info(f"Building context string from {len(df)} search results")
    chunks = []
    for _, row in df.iterrows():
        text = row["text"].strip()
        meta = row["metadata"] or {}
        fn    = meta.get("filename", "Unknown file")
        pages = list(meta.get("page_numbers", []))
        # pages = list(meta.get("page_numbers") or [])  # pages = meta.get("page_numbers", [])
        title = meta.get("title", "")
        src   = fn + (f" p. {', '.join(map(str, pages))}" if pages else "")
        dist  = row.get("_distance", None)
        sim   = f"{1 - dist:.3f}" if dist is not None else "N/A"

        part = f"{text}\nSource: {src}"
        if title:
            part += f"\nTitle: {title}"
        part += f"\nSimilarity: {sim}"
        chunks.append(part)
    
    context_length = len("\n\n".join(chunks))
    logger.info(f"Built context string with {context_length} characters")
    return "\n\n".join(chunks)

def get_chat_response(messages: list, context: str) -> str:
    """Stream a chat response from GPT-4o-mini, injecting the context."""
    system = f"""You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context. If the context doesn't contain the relevant information, say so.

Context:
{context}
"""
    full = [{"role": "system", "content": system}, *messages]
    
    try:
        logger.info("Generating chat response with OpenAI API")
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full,
            temperature=0.3,
            stream=True,
        )
        logger.info("Successfully created OpenAI stream")
        return st.write_stream(stream)
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating the response."

# --- Streamlit App ---
st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")

logger.info("Starting Streamlit Document Q&A System")

# Header with logo
try:
    st.image("misc/bmrn.jpeg", width=300)
except:
    logger.warning("Logo file misc/bmrn.jpeg not found")

st.title("Index Q&A - System:")

# Sidebar: DB status
with st.sidebar:
    st.header("📊 Database Status")
    table = init_db()
    if table:
        try:
            cnt = table.count_rows()
            logger.info(f"Database connected successfully with {cnt} rows")
            st.success("✅ Connected")
            st.info(f"📄 {cnt:,} chunks indexed")
            with st.expander("Schema Details"):
                for fld in table.schema:
                    st.text(f"• {fld.name}: {fld.type}")
        except Exception as e:
            logger.error(f"Error reading table: {e}")
            st.error(f"Error reading table: {e}")
            table = None
    else:
        logger.error("Database connection failed")
        st.error("❌ Database not connected")

if not table:
    st.stop()

# CSS for results
st.markdown(
    """
    <style>
    .search-result {margin:10px 0;padding:10px;border-radius:4px;background:#f0f2f6;}
    .search-result summary {cursor:pointer;color:#0f52ba;font-weight:500;}
    .search-result summary:hover {color:#1e90ff;}
    .metadata {font-size:0.9em;color:#666;font-style:italic;margin-bottom:4px;}
    .content-text {margin-top:8px;line-height:1.4;color:#333;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User question
if prompt := st.chat_input("Ask a question about your documents..."):
    logger.info(f"User submitted query: '{prompt}'")
    
    # Show user
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 1) Search DB
    with st.status("🔍 Searching documents...", expanded=False) as status:
        df = search_docs(prompt, table, num_results=5)
        
        if not df.empty:
            logger.info(f"Search completed successfully, found {len(df)} results")
            status.update(label=f"✅ Found {len(df)} relevant sections", state="complete")
        else:
            logger.warning("Search completed but no results found")
            status.update(label="⚠️ No relevant sections found", state="complete")

    # 2) Display results
    if df is not None and not df.empty:
        st.write("📋 **Found relevant sections:**")
        for _, row in df.iterrows():
            text = row["text"].strip()
            meta = row["metadata"] or {}
            fn    = meta.get("filename", "Unknown file")
            pages = list(meta.get("page_numbers", []))
            # pages = list(meta.get("page_numbers") or [])  # pages = meta.get("page_numbers", [])
            title = meta.get("title", "Untitled section")
            src   = fn + (f" p. {', '.join(map(str, pages))}" if pages else "")
            dist  = row.get("_distance", None)
            sim   = f"{1 - dist:.3f}" if dist is not None else "N/A"

            st.markdown(
                f"""
                <div class="search-result">
                  <details>
                    <summary>📄 {src} (Similarity: {sim})</summary>
                    <div class="metadata">📖 Section: {title}</div>
                    <div class="content-text">{text}</div>
                  </details>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("⚠️ No relevant content found for your query.")

    # 3) Build context string for LLM
    context = build_context_string(df) if df is not None else ""

    # 4) Get assistant reply
    with st.chat_message("assistant"):
        logger.info("Generating assistant response")
        reply = get_chat_response(st.session_state.messages, context)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    logger.info("Assistant response completed and added to chat history")
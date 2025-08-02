import os
import sys
import argparse
from typing import List

import streamlit as st
import pandas as pd
import lancedb
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

import os, sys
sys.path.append("../")
from docling.document_converter import DocumentConverter
from bin.sitemap import get_sitemap_urls
from typing import List
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from openai import OpenAI
from bin.tokenizer import OpenAITokenizerWrapper

# ---------------------------
# Configuration
# ---------------------------
DB_URI = os.getenv("LANCEDB_URI", "data/lancedb")
TABLE_NAME = os.getenv("LANCEDB_TABLE", "lance_neur_papers_db")
DEFAULT_URLS = [
    "https://arxiv.org/pdf/2408.09869",
    # add more URLs or file paths here
]
MAX_TOKENS = 8191
MODEL_NAME = "gpt-4o-mini"

# ---------------------------
# Ingestion
# ---------------------------

def ingest_documents(urls: List[str], db_uri: str, table_name: str):
    """
    Convert, chunk, embed, and store documents into a LanceDB table.
    """
    load_dotenv()
    converter = DocumentConverter()
    # Convert URLs into docling documents
    results = converter.convert_all(urls)
    documents = [res.document for res in results if res.document]

    # Initialize OpenAI tokenizer wrapper
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model("gpt-4o"),
        max_tokens=MAX_TOKENS,
    )
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS)

    # Chunk documents
    processed_chunks = []
    for doc in documents:
        for chunk in chunker.chunk(dl_doc=doc):
            meta = chunk.meta
            pages = sorted({prov.page_no for item in meta.doc_items for prov in item.prov})
            processed_chunks.append({
                "text": chunk.text,
                "metadata": {
                    "filename": meta.origin.filename,
                    "page_numbers": pages or None,
                    "title": meta.headings[0] if meta.headings else None,
                },
            })

    # Connect to LanceDB and create table
    db = lancedb.connect(db_uri)
    embed_func = get_registry().get("openai").create(name="text-embedding-3-small")

    class ChunkMetadata(LanceModel):
        filename: str | None
        page_numbers: List[int] | None
        title: str | None

    class ChunkRecord(LanceModel):
        text: str = embed_func.SourceField()
        vector: Vector(embed_func.ndims()) = embed_func.VectorField()  # type: ignore
        metadata: ChunkMetadata

    table = db.create_table(table_name, schema=ChunkRecord, mode="overwrite")
    table.add(processed_chunks)
    print(f"Ingested {table.count_rows():,} chunks into '{table_name}' at '{db_uri}'")


# ---------------------------
# Streamlit App
# ---------------------------

@st.cache_resource
def init_table():
    """Connect to existing LanceDB table (cached)."""
    db = lancedb.connect(DB_URI)
    return db.open_table(TABLE_NAME)


def search_docs(query: str, table, num_results: int = 5) -> pd.DataFrame:
    """Perform vector search and return flattened DataFrame."""
    return (
        table
        .search(query=query, query_type="vector")
        .limit(num_results)
        .select_columns(["text", "metadata"])
        .to_pandas(flatten=True)
    )


def build_context_string(df: pd.DataFrame) -> str:
    """Build a context string for the LLM from search results."""
    contexts = []
    for _, row in df.iterrows():
        text = row.get("text", "").strip()
        meta = row.get("metadata") or {}
        filename = meta.get("filename", "")
        pages = meta.get("page_numbers", [])
        title = meta.get("title", "")
        parts = []
        if filename:
            parts.append(filename)
        if pages:
            parts.append(f"p. {', '.join(str(p) for p in pages)}")
        source = " - ".join(parts) or "Unknown source"
        chunk = text + f"\nSource: {source}"
        if title:
            chunk += f"\nTitle: {title}"
        contexts.append(chunk)
    return "\n\n".join(contexts)


def get_chat_response(messages, context: str) -> str:
    """Stream a chat response from the LLM using provided context."""
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer questions. If you're unsure or the context
doesn't contain the relevant information, say so.

Context:
{context}
"""
    full_messages = [{"role": "system", "content": system_prompt}, *messages]
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=full_messages,
        temperature=0.7,
        stream=True,
    )
    return st.write_stream(stream)


def run_app():
    """Launch the Streamlit Q&A interface."""
    st.set_page_config(page_title="📚 Document Q&A", page_icon="📚", layout="wide")
    st.title("📚 Document Q&A")
    # CSS styling
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

    table = init_table()

    # Sidebar status
    with st.sidebar:
        st.header("📊 Database Status")
        try:
            cnt = table.count_rows()
            st.success("✅ Connected")
            st.info(f"📄 {cnt:,} chunks indexed")
            with st.expander("Schema Details"):
                for fld in table.schema:
                    st.text(f"• {fld.name}: {fld.type}")
        except Exception as e:
            st.error(f"❌ DB error: {e}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about the document..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})

        # Search for context
        with st.status("🔍 Searching document...", expanded=False):
            df = search_docs(prompt, table, num_results=3)
            if not df.empty:
                st.write("📋 **Found relevant sections:**")
                for _, row in df.iterrows():
                    txt = row.get("text", "").strip()
                    md = row.get("metadata") or {}
                    fname = md.get("filename", "Unknown file")
                    pages = md.get("page_numbers", [])
                    title = md.get("title", "Untitled section")
                    page_str = f"p. {', '.join(str(p) for p in pages)}" if pages else ""
                    source = f"{fname} {page_str}".strip()
                    st.markdown(
                        f"""
                        <div class="search-result">
                          <details>
                            <summary>📄 {source}</summary>
                            <div class="metadata">📖 Section: {title}</div>
                            <div class="content-text">{txt}</div>
                          </details>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                context_text = build_context_string(df)
            else:
                st.warning("⚠️ No relevant content found.")
                context_text = ""

        # Assistant response
        with st.chat_message("assistant"):
            reply = get_chat_response(st.session_state.messages, context_text)
        st.session_state.messages.append({"role":"assistant","content":reply})


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI()

    parser = argparse.ArgumentParser(description="Doc Q&A app with optional ingestion.")
    parser.add_argument(
        "--ingest", action="store_true", help="Ingest documents into LanceDB and exit"
    )
    parser.add_argument(
        "--urls", nargs="+", help="List of URLs or file paths to ingest"
    )
    args = parser.parse_args()

    if args.ingest:
        urls = args.urls or DEFAULT_URLS
        ingest_documents(urls, DB_URI, TABLE_NAME)
    else:
        run_app()

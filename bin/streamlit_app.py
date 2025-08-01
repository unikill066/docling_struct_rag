# app.py

import os, json
from dotenv import load_dotenv
from pathlib import Path
from tempfile import mkdtemp
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# Load .env vars
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EXPORT_TYPE = ExportType.DOC_CHUNKS
COLLECTION_NAME = "docling_demo"
MILVUS_DB_PATH = Path("docling.db")

# UI
st.title("📚 Doc Ingestion with LangChain + Milvus")
uploaded_file = st.file_uploader("Upload a PDF to ingest", type="pdf")

if uploaded_file:
    # Save the uploaded file
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    save_path = uploads_dir / uploaded_file.name

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Saved file to: `{save_path}`")

    if st.button("🚀 Ingest into Vectorstore"):
        with st.spinner("Loading and chunking..."):
            # Set up embedding
            embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

            # Load document
            loader = DoclingLoader(
                file_path=[str(save_path)],
                export_type=EXPORT_TYPE,
                chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
            )
            docs = loader.load()
            splits = docs  # already chunked from Docling

            # Ingest into Milvus
            milvus_uri = str(MILVUS_DB_PATH)
            vectorstore = Milvus.from_documents(
                documents=splits,
                embedding=embedding,
                collection_name=COLLECTION_NAME,
                connection_args={"uri": milvus_uri},
                index_params={"index_type": "FLAT"},
                drop_old=False,  # keep previous data
            )

        st.success("✅ Ingestion complete! Vectorstore updated.")

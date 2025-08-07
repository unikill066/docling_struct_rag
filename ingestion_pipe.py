from docling.document_converter import DocumentConverter
from lancedb import connect
from typing import List
from logger import get_logger
from config import LANCEDB_PATH, TABLE_NAME
from chunking_utils import create_chunker
from embedding_utils import Chunks
from openai import OpenAI
from dotenv import load_dotenv

log = get_logger()

def extract_documents(sources: List[str]):
    log.info("Starting document conversion...")
    converter = DocumentConverter()
    results = converter.convert_all(sources)
    documents = [res.document for res in results if res.document]
    log.info(f"Converted {len(documents)} documents.")
    return documents

def chunk_documents(documents):
    chunker = create_chunker()
    all_chunks = []
    for doc in documents:
        chunks = list(chunker.chunk(dl_doc=doc))
        log.info(f"Chunked document into {len(chunks)} chunks.")
        all_chunks.extend(chunks)
    return all_chunks

def prepare_chunks(chunk_list):
    log.info("Preparing chunks for embedding...")
    return [
        {
            "text": chunk.text,
            "metadata": {
                "filename": chunk.meta.origin.filename,
                "page_numbers": [
                page_no
                for page_no in sorted(
                    set(
                        prov.page_no
                        for item in chunk.meta.doc_items
                        for prov in item.prov
                    )
                )
            ]
            or None,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            },
        }
        for chunk in chunk_list
    ]

def embed_and_store(processed_chunks):
    log.info("Storing chunks in LanceDB...")
    db = connect(LANCEDB_PATH)
    table = db.create_table(TABLE_NAME, schema=Chunks, mode="overwrite")
    table.add(processed_chunks)
    log.info(f"Stored {table.count_rows()} rows.")
    log.info(table.to_pandas())
    return table
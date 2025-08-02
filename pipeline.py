from ingestion_pipe import extract_documents, chunk_documents, prepare_chunks, embed_and_store

if __name__ == "__main__":
    sources = [
        "https://arxiv.org/pdf/2408.09869",
        "https://arxiv.org/pdf/2408.09869",
        ""

    ]
    documents = extract_documents(sources)
    chunks = chunk_documents(documents)
    processed_chunks = prepare_chunks(chunks)
    embed_and_store(processed_chunks)

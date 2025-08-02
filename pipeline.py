from ingestion_pipe import extract_documents, chunk_documents, prepare_chunks, embed_and_store

import os
print(os.listdir("/Users/discovery/Desktop/Docling/data"))

if __name__ == "__main__":
    sources = [
        "data/epigenomic_landscape_of_the_human_dorsal_root.16.pdf",
        "data/persistent_changes_in_the_dorsal_root_ganglion.18.pdf",
        "data/180190.1-20250520113907-covered-e0fd13ba177f913fd3156f593ead4cfd.pdf",
        "data/NIHPP-2024.06.15.599167v1.pdf",
        "data/2024.12.20.629638v1.full.pdf",
        "data/2025.03.24.645122v1.full.pdf",
        "data/2025.03.26.645611v1.full.pdf",
    ]
    documents = extract_documents(sources)
    chunks = chunk_documents(documents)
    processed_chunks = prepare_chunks(chunks)
    embed_and_store(processed_chunks)
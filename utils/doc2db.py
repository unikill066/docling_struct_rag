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

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from bin.tokenizer import OpenAITokenizerWrapper

converter = DocumentConverter()

# # --------------------------------------------------------------
# # Basic PDF extraction
# # --------------------------------------------------------------

# result = converter.convert("https://arxiv.org/pdf/2408.09869")

# document = result.document
# markdown_output = document.export_to_markdown()
# json_output = document.export_to_dict()

# print(markdown_output)

# # --------------------------------------------------------------
# # Basic HTML extraction
# # --------------------------------------------------------------

# result = converter.convert("https://research.ibm.com/blog/ai-deep-search-docqa")

# document = result.document
# markdown_output = document.export_to_markdown()
# print(markdown_output)

# # --------------------------------------------------------------
# # Scrape multiple pages using the sitemap
# # --------------------------------------------------------------

# sitemap_urls = get_sitemap_urls("https://research.ibm.com/blog/ai-deep-search-docqa")
# conv_results_iter = converter.convert_all(sitemap_urls)

# docs = []
# for result in conv_results_iter:
#     if result.document:
#         document = result.document
#         docs.append(document)


result = converter.convert_all(["https://arxiv.org/pdf/2408.09869", "https://arxiv.org/pdf/2408.09869"])
    
#     ["../data/2025.03.26.645611v1.full.pdf",
# "../data/2024.12.20.629638v1.full.pdf",])
# "../data/persistent_changes_in_the_dorsal_root_ganglion.18.pdf",
# "../data/epigenomic_landscape_of_the_human_dorsal_root.16.pdf"])
# print(result)
print("DONE")

documents = []
for result in result:
    if result.document:
        document = result.document
        documents.append(document)

# document = result.document
# markdown_output = document.export_to_markdown()
# json_output = document.export_to_dict()

# print(markdown_output)


load_dotenv()

# Initialize OpenAI client (make sure you have OPENAI_API_KEY in your environment variables)
client = OpenAI()


import tiktoken

from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
MAX_TOKENS = 8191
tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.encoding_for_model("gpt-4o"),
    max_tokens=MAX_TOKENS,
)


chunker = HybridChunker(
    tokenizer=tokenizer,
    max_tokens=MAX_TOKENS,
    # merge_peers=True,
)

chunk_list = list()
for doc in documents:
    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)
    print(len(chunks))
    chunk_list.extend(chunks)



# Lance DB
db = lancedb.connect("../data/lancedb")
func = get_registry().get("openai").create(name="text-embedding-3-small")

# Define a simplified metadata schema
class ChunkMetadata(LanceModel):
    """
    You must order the fields in alphabetical order.
    This is a requirement of the Pydantic implementation.
    """

    filename: str | None
    page_numbers: List[int] | None
    title: str | None


# Define the main Schema
class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    metadata: ChunkMetadata


table = db.create_table("lance_neur_papers_db", schema=Chunks, mode="overwrite")

# --------------------------------------------------------------
# Prepare the chunks for the table
# --------------------------------------------------------------

# Create table with processed chunks
processed_chunks = [
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

# --------------------------------------------------------------
# Add the chunks to the table (automatically embeds the text)
# --------------------------------------------------------------

table.add(processed_chunks)

# --------------------------------------------------------------
# Load the table
# --------------------------------------------------------------

table.to_pandas()
table.count_rows()
import json, os
import logging
import time
from collections.abc import Iterable
from pathlib import Path
import yaml

from dotenv import load_dotenv

from docling_core.types.doc import ImageRefMode
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from docling.chunking import HybridChunker

# ────────────────────────────────────────────────────────────────────────────────
# Load environment and set logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
USE_V2 = True
USE_LEGACY = False
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "docling_batch_v2"
MILVUS_DB_PATH = Path(os.getcwd()) / "docling_batch_v2.db"

# ────────────────────────────────────────────────────────────────────────────────
def export_documents(conv_results: Iterable[ConversionResult], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            if USE_V2:
                conv_res.document.save_as_json(output_dir / f"{doc_filename}.json", image_mode=ImageRefMode.PLACEHOLDER)
                conv_res.document.save_as_html(output_dir / f"{doc_filename}.html", image_mode=ImageRefMode.EMBEDDED)
                conv_res.document.save_as_document_tokens(output_dir / f"{doc_filename}.doctags.txt")
                conv_res.document.save_as_markdown(output_dir / f"{doc_filename}.md", image_mode=ImageRefMode.PLACEHOLDER)
                conv_res.document.save_as_markdown(output_dir / f"{doc_filename}.txt", image_mode=ImageRefMode.PLACEHOLDER, strict_text=True)
                with (output_dir / f"{doc_filename}.yaml").open("w") as fp:
                    fp.write(yaml.safe_dump(conv_res.document.export_to_dict()))
                with (output_dir / f"{doc_filename}.doctags.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_document_tokens())
                with (output_dir / f"{doc_filename}.md").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown())
                with (output_dir / f"{doc_filename}.txt").open("w") as fp:
                    fp.write(conv_res.document.export_to_markdown(strict_text=True))

            if USE_LEGACY:
                with (output_dir / f"{doc_filename}.legacy.json").open("w", encoding="utf-8") as fp:
                    fp.write(json.dumps(conv_res.legacy_document.export_to_dict()))
                with (output_dir / f"{doc_filename}.legacy.txt").open("w", encoding="utf-8") as fp:
                    fp.write(conv_res.legacy_document.export_to_markdown(strict_text=True))
                with (output_dir / f"{doc_filename}.legacy.md").open("w", encoding="utf-8") as fp:
                    fp.write(conv_res.legacy_document.export_to_markdown())
                with (output_dir / f"{doc_filename}.legacy.doctags.txt").open("w", encoding="utf-8") as fp:
                    fp.write(conv_res.legacy_document.export_to_document_tokens())

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.info(f"Document {conv_res.input.file} was partially converted with the following errors:")
            for item in conv_res.errors:
                _log.info(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.info(f"Processed {success_count + partial_success_count + failure_count} docs, "
              f"of which {failure_count} failed "
              f"and {partial_success_count} were partially converted.")
    return success_count, partial_success_count, failure_count

# ────────────────────────────────────────────────────────────────────────────────
def main():
    # Define your input files
    input_doc_paths = [
        Path("data/2025.03.26.645611v1.full.pdf"),
        # Path("data/180190.1-20250520113907-covered-e0fd13ba177f913fd3156f593ead4cfd.pdf"),
        # Path("data/persistent_changes_in_the_dorsal_root_ganglion.18.pdf"),
        # Path("data/nihpp-2024.06.15.599167v1.pdf"),
        Path("data/2024.12.20.629638v1.full.pdf"),
        Path("data/epigenomic_landscape_of_the_human_dorsal_root.16.pdf"),
        # Path("data/2025.03.24.645122v1.full.pdf"),
        # Path("data/PIIS1526590024001354.pdf"),
    ]

    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_page_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseV4DocumentBackend,
            )
        }
    )

    start_time = time.time()

    # Convert documents
    conv_results = doc_converter.convert_all(input_doc_paths, raises_on_error=False)

    # Export to files
    output_dir = Path("scratch")
    success_count, partial_success_count, failure_count = export_documents(conv_results, output_dir)

    # Chunk and prepare for vectorstore
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    chunker = HybridChunker(tokenizer=EMBED_MODEL_ID)

    all_chunks = []
    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            chunks = chunker.chunk(conv_res.document)
            all_chunks.extend([Document(page_content=chunk.text) for chunk in chunks])

    # Store in Milvus Lite
    vectorstore = Milvus.from_documents(
        documents=all_chunks,
        embedding=embedding,
        collection_name=COLLECTION_NAME,
        connection_args={"uri": str(MILVUS_DB_PATH)},
        index_params={"index_type": "FLAT"},
        drop_old=True,
    )

    end_time = time.time() - start_time
    _log.info(f"✅ Document conversion + ingestion complete in {end_time:.2f} seconds.")

    if failure_count > 0:
        raise RuntimeError(f"The batch failed on {failure_count} out of {len(input_doc_paths)} files.")

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

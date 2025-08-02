from typing import List
from pydantic import BaseModel
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from config import EMBEDDING_MODEL

class ChunkMetadata(LanceModel):
    filename: str | None
    page_numbers: List[int] | None
    title: str | None

class Chunks(LanceModel):
    text: str = get_registry().get("openai").create(name=EMBEDDING_MODEL).SourceField()
    vector: Vector(get_registry().get("openai").create(name=EMBEDDING_MODEL).ndims()) = \
        get_registry().get("openai").create(name=EMBEDDING_MODEL).VectorField()
    metadata: ChunkMetadata

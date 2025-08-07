from typing import List
from pydantic import BaseModel
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from config import EMBEDDING_MODEL

embed_fcn = get_registry().get("openai").create(name=EMBEDDING_MODEL)

class ChunkMetadata(LanceModel):
    filename: str | None
    page_numbers: List[int] | None
    title: str | None

class Chunks(LanceModel):
    text: str = embed_fcn.SourceField()
    vector: Vector(embed_fcn.ndims()) = embed_fcn.VectorField(default=None)
    metadata: ChunkMetadata
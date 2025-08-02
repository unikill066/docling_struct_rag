import os
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 8191
MODEL_NAME = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
LANCEDB_PATH = "data/lancedb"
TABLE_NAME = "lance_neur_papers_db"

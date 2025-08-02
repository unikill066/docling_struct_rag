import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling.chunking import HybridChunker
from config import MAX_TOKENS, MODEL_NAME

def create_chunker():
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model(MODEL_NAME),
        max_tokens=MAX_TOKENS,
    )
    return HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS)

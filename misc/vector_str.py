import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain 

# ────────────────────────────────────────────────────────────────────────────────
# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ────────────────────────────────────────────────────────────────────────────────
# Config
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MILVUS_DB_PATH = Path(os.getcwd()) / "docling_batch_v2.db"
COLLECTION_NAME = "docling_batch_v2"
TOP_K = 3
QUESTION = "What are some methodologies in neuropathic pain?"
# ────────────────────────────────────────────────────────────────────────────────
# Load embeddings and vectorstore
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

vectorstore = Milvus(
    embedding_function=embedding,
    collection_name="docling_batch_v2",
    connection_args={"uri": "docling_batch_v2.db"},
)



# ────────────────────────────────────────────────────────────────────────────────
# Create retriever and LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# Option: use OpenAI
llm = ChatOpenAI(
    model="o4-mini",  # Change if using GPT-4 or other
    use_previous_response_id=True,
)

# Option: use HuggingFace instead (uncomment below)
# from langchain_huggingface import HuggingFaceEndpoint
# HF_TOKEN = os.getenv("HF_TOKEN")
# GEN_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# llm = HuggingFaceEndpoint(repo_id=GEN_MODEL_ID, huggingfacehub_api_token=HF_TOKEN)

# ────────────────────────────────────────────────────────────────────────────────
# Prompt
PROMPT = PromptTemplate.from_template(
    "Context information is below.\n---------------------\n{context}\n---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {input}\nAnswer:\n"
)

# ────────────────────────────────────────────────────────────────────────────────
# RAG chain setup
question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ────────────────────────────────────────────────────────────────────────────────
# Ask the question
resp_dict = rag_chain.invoke({"input": QUESTION})

# ────────────────────────────────────────────────────────────────────────────────
# Display the response
def clip_text(text, threshold=100):
    return f"{text[:threshold]}..." if len(text) > threshold else text

clipped_answer = clip_text(resp_dict["answer"], threshold=200)
print(f"📌 Question:\n{resp_dict['input']}\n\n✅ Answer:\n{clipped_answer}")

for i, doc in enumerate(resp_dict["context"]):
    print()
    print(f"📄 Source {i + 1}:")
    print(f"  text: {json.dumps(clip_text(doc.page_content, threshold=350))}")
    for key in doc.metadata:
        if key != "pk":
            val = doc.metadata.get(key)
            clipped_val = clip_text(val) if isinstance(val, str) else val
            print(f"  {key}: {clipped_val}")

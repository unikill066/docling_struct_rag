import streamlit as st
import lancedb
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load env (OPENAI_API_KEY, etc.)
load_dotenv()
client = OpenAI()

@st.cache_resource
def init_db():
    """Connect (and cache) the LanceDB table."""
    try:
        db = lancedb.connect("/Users/discovery/Desktop/Docling/data/lancedb")
        return db.open_table("lance_neur_papers_db")
    except Exception as e:
        st.error(f"DB connection failed: {e}")
        return None

def search_docs(query: str, table, num_results: int = 5) -> pd.DataFrame:
    """
    Vector-search the table, return a DataFrame with text, metadata, and _distance.
    We omit flatten so metadata remains a dict.
    """
    df = (
        table
        .search(query=query, query_type="vector")
        .limit(num_results)
        .to_pandas()  # no flatten arg
    )
    return df.loc[:, ["text", "metadata", "_distance"]]

def build_context_string(df: pd.DataFrame) -> str:
    """Build a single context string for the LLM from the search results."""
    chunks = []
    for _, row in df.iterrows():
        text = row["text"].strip()
        meta = row["metadata"] or {}
        fn    = meta.get("filename", "Unknown file")
        pages = meta.get("page_numbers", [])
        title = meta.get("title", "")
        src   = fn + (f" p. {', '.join(map(str, pages))}" if pages else "")
        dist  = row.get("_distance", None)
        sim   = f"{1 - dist:.3f}" if dist is not None else "N/A"

        part = f"{text}\nSource: {src}"
        if title:
            part += f"\nTitle: {title}"
        part += f"\nSimilarity: {sim}"
        chunks.append(part)
    return "\n\n".join(chunks)

def get_chat_response(messages: list, context: str) -> str:
    """Stream a chat response from GPT-4o-mini, injecting the context."""
    system = f"""You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context. If the context doesn't contain the relevant information, say so.

Context:
{context}
"""
    full = [{"role": "system", "content": system}, *messages]
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=full,
        temperature=0.3,
        stream=True,
    )
    return st.write_stream(stream)

# --- Streamlit App ---
st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")
st.image("misc/bmrn.jpeg", width=300)
st.title("Index Q&A - System:")



# Sidebar: DB status
with st.sidebar:
    st.header("📊 Database Status")
    table = init_db()
    if table:
        try:
            cnt = table.count_rows()
            st.success("✅ Connected")
            st.info(f"📄 {cnt:,} chunks indexed")
            with st.expander("Schema Details"):
                for fld in table.schema:
                    st.text(f"• {fld.name}: {fld.type}")
        except Exception as e:
            st.error(f"Error reading table: {e}")
            table = None
    else:
        st.error("❌ Database not connected")

if not table:
    st.stop()

# CSS for results
st.markdown(
    """
    <style>
    .search-result {margin:10px 0;padding:10px;border-radius:4px;background:#f0f2f6;}
    .search-result summary {cursor:pointer;color:#0f52ba;font-weight:500;}
    .search-result summary:hover {color:#1e90ff;}
    .metadata {font-size:0.9em;color:#666;font-style:italic;margin-bottom:4px;}
    .content-text {margin-top:8px;line-height:1.4;color:#333;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User question
if prompt := st.chat_input("Ask a question about your documents..."):
    # Show user
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 1) Search DB
    df = search_docs(prompt, table, num_results=5)

    # 2) Display results
    if df is not None and not df.empty:
        st.write("📋 **Found relevant sections:**")
        for _, row in df.iterrows():
            text = row["text"].strip()
            meta = row["metadata"] or {}
            fn    = meta.get("filename", "Unknown file")
            pages = meta.get("page_numbers", [])
            title = meta.get("title", "Untitled section")
            src   = fn + (f" p. {', '.join(map(str, pages))}" if pages else "")
            dist  = row.get("_distance", None)
            sim   = f"{1 - dist:.3f}" if dist is not None else "N/A"

            st.markdown(
                f"""
                <div class="search-result">
                  <details>
                    <summary>📄 {src} (Similarity: {sim})</summary>
                    <div class="metadata">📖 Section: {title}</div>
                    <div class="content-text">{text}</div>
                  </details>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("⚠️ No relevant content found for your query.")

    # 3) Build context string for LLM
    context = build_context_string(df) if df is not None else ""

    # 4) Get assistant reply
    with st.chat_message("assistant"):
        reply = get_chat_response(st.session_state.messages, context)
    st.session_state.messages.append({"role": "assistant", "content": reply})



# import streamlit as st
# import lancedb
# from openai import OpenAI
# from dotenv import load_dotenv
# import pandas as pd

# load_dotenv()

# client = OpenAI()

# @st.cache_resource
# def init_db():
#     """Initialize database connection"""
#     try:
#         db = lancedb.connect("/Users/discovery/Desktop/Docling/data/lancedb")
#         table = db.open_table("lance_neur_papers_db")
#         return table
#     except Exception as e:
#         st.error(f"Failed to connect to database: {str(e)}")
#         return None

# def get_context(query: str, table, num_results: int = 5) -> str:
#     try:
#         # Simple search - no fancy stuff
#         results = table.search(query=query, query_type="vector").limit(num_results).to_df()
#         print(results)
#         return results
#     #     if results.empty:
#     #         return "No results found."
        
#     #     contexts = []
#     #     for _, row in results.iterrows():
#     #         text = str(row.get('text', ''))
#     #         if text.strip():
#     #             contexts.append(text)
        
#     #     return "\n\n".join(contexts)
    
#     except Exception as e:
#         st.error(f"Search failed: {e}")
#         return "Search error - check console for details."

# def get_chat_response(messages, context: str) -> str:
#     """Generate chat response with context"""
#     system_prompt = f"""You are a helpful assistant that answers questions based on the provided context from technical documents.
    
#     Instructions:
#     1. Use only the information from the provided context to answer questions
#     2. If the context doesn't contain relevant information, say so clearly
#     3. When referencing information, mention the source when available
#     4. Be precise and technical when appropriate
#     5. If multiple sources provide different information, note the discrepancies
    
#     Context:
#     {context}
#     """

#     messages_with_context = [{"role": "system", "content": system_prompt}, *messages]

#     try:
#         # Create the streaming response
#         stream = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages_with_context,
#             temperature=0.3,  # Lower temperature for more factual responses
#             stream=True,
#         )

#         # Use Streamlit's built-in streaming capability
#         response = st.write_stream(stream)
#         return response
#     except Exception as e:
#         st.error(f"Error generating response: {str(e)}")
#         return "Sorry, I encountered an error while generating the response."




# #### starts from here

# # Initialize Streamlit app
# st.set_page_config(
#     page_title="Document Q&A",
#     page_icon="📚",
#     layout="wide"
# )
# st.image("misc/bmrn.jpeg", width=200)

# st.title("📚 Document Q&A System")
# st.markdown("Ask questions about your indexed documents and get contextual answers.")

# # Sidebar with database info
# with st.sidebar:
#     st.header("📊 Database Status")
    
#     # Initialize database connection
#     table = init_db()
    
#     if table is not None:
#         try:
#             row_count = table.count_rows()
#             st.success(f"✅ Connected")
#             st.info(f"📄 {row_count:,} document chunks indexed")
            
#             # Show schema info
#             with st.expander("Schema Details"):
#                 try:
#                     schema = table.schema
#                     for field in schema:
#                         st.text(f"• {field.name}: {field.type}")
#                 except:
#                     st.text("Schema information not available")
                    
#         except Exception as e:
#             st.error(f"❌ Database error: {str(e)}")
#             table = None
#     else:
#         st.error("❌ Database not connected")
#         st.info("Please check your database setup and try again.")

# # Main chat interface
# if table is not None:
#     # Initialize session state for chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display chat messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Chat input
#     if prompt := st.chat_input("Ask a question about your documents..."):
#         # Display user message
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})

#         # Get relevant context
#         with st.status("🔍 Searching documents...", expanded=False):
#             # only load the columns you need, use to_pandas(flatten=True)
#             context = (
#                 table
#                 .search(query=prompt, query_type="vector")
#                 .limit(5)
#                 .select_columns(["text","metadata","_distance"])
#                 .to_pandas(flatten=True)
#             )

#         # # Get relevant context
#         # with st.status("🔍 Searching documents...", expanded=False) as status:
#         #     # context = get_context(prompt, table)
#         #     context = table.search(query=prompt, query_type="vector").limit(5).to_df()

#             print(type(context))
#             print(context)
            
#             # Add custom CSS for search results
#             st.markdown(
#                 """
#                 <style>
#                 .search-result {
#                     margin: 10px 0;
#                     padding: 15px;
#                     border-radius: 8px;
#                     background-color: #f8f9fa;
#                     border-left: 4px solid #0f52ba;
#                 }
#                 .search-result summary {
#                     cursor: pointer;
#                     color: #0f52ba;
#                     font-weight: 600;
#                     font-size: 0.95em;
#                 }
#                 .search-result summary:hover {
#                     color: #1e90ff;
#                 }
#                 .metadata {
#                     font-size: 0.85em;
#                     color: #666;
#                     font-style: italic;
#                     margin: 5px 0;
#                 }
#                 .content-text {
#                     margin-top: 10px;
#                     line-height: 1.5;
#                     color: #333;
#                 }
#                 </style>
#             """,
#                 unsafe_allow_html=True,
#             )

#             # if context and context != "No context found due to search error.":
#             if context is not None and not context.empty:
#                 st.write("📋 **Found relevant sections:**")
                
#                 for i, row in context.iterrows():
#                     # Extract text
#                     text = row["text"].strip()

#                     # Extract metadata dict
#                     meta = row.get("metadata") or {}
#                     filename = meta.get("filename", "Unknown source")
#                     pages = meta.get("page_numbers", [])
#                     title = meta.get("title", "Untitled section")

#                     # Build a nice “Source” line
#                     page_str = f"p. {', '.join(str(p) for p in pages)}" if pages else ""
#                     source = f"{filename} {page_str}".strip()

#                     # Compute similarity if _distance is present
#                     dist = row.get("_distance", None)
#                     similarity = f"{1 - dist:.3f}" if dist is not None else "N/A"

#                     # Now render in Streamlit
#                     st.markdown(
#                         f"""
#                         <div class="search-result">
#                         <details>
#                             <summary>📄 {source} (Similarity: {similarity})</summary>
#                             <div class="metadata">📖 Section: {title}</div>
#                             <div class="content-text">{text}</div>
#                         </details>
#                         </div>
#                         """,
#                         unsafe_allow_html=True,
#                     )
#             else:
#                 st.warning("⚠️ No relevant content found for your query.")

#         # Display assistant response
#         with st.chat_message("assistant"):
#             response = get_chat_response(st.session_state.messages, context)

#         # Add assistant response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": response})

# else:
#     st.error("Cannot start Q&A system without database connection.")
#     st.info("Please check your database setup and restart the application.")

# # Footer
# st.markdown("---")
# st.markdown("💡 **Tip:** Ask specific questions about the documents for better results!")
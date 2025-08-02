import streamlit as st
import lancedb
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

@st.cache_resource
def init_table():
    """Initialize and cache the LanceDB table connection."""
    db = lancedb.connect("data/lancedb")
    return db.open_table("lance_neur_papers_db")

def search_docs(query: str, table, num_results: int = 5) -> pd.DataFrame:
    """
    Vector‐search the table and return a small flattened DataFrame
    containing just the text and metadata columns.
    """
    return (
        table
        .search(query=query, query_type="vector")
        .limit(num_results)
        .select_columns(["text", "metadata"])
        .to_pandas(flatten=True)
    )

def build_context_string(df: pd.DataFrame) -> str:
    """
    Turn the search‐result DataFrame into a single string for the LLM,
    with “Source:” and “Title:” for each chunk.
    """
    contexts = []
    for _, row in df.iterrows():
        text = row.get("text", "").strip()
        meta = row.get("metadata") or {}
        filename = meta.get("filename", "")
        pages = meta.get("page_numbers", [])
        title = meta.get("title", "")
        # build source line
        parts = []
        if filename:
            parts.append(filename)
        if pages:
            parts.append(f"p. {', '.join(str(p) for p in pages)}")
        source = " - ".join(parts) or "Unknown source"

        chunk = text + f"\nSource: {source}"
        if title:
            chunk += f"\nTitle: {title}"
        contexts.append(chunk)
    return "\n\n".join(contexts)

def get_chat_response(messages, context: str) -> str:
    """Stream a chat response, injecting our retrieved context."""
    system_prompt = f"""You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer questions. If you're unsure or the context
doesn't contain the relevant information, say so.

Context:
{context}
"""
    full = [{"role":"system", "content": system_prompt}, *messages]
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=full,
        temperature=0.7,
        stream=True,
    )
    return st.write_stream(stream)

# --- Streamlit page setup ---
st.set_page_config(page_title="📚 Document Q&A", page_icon="📚", layout="wide")
st.title("📚 Document Q&A")

# Inject CSS for result styling
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

# Load (and cache) the LanceDB table
table = init_table()

# Sidebar: database status
with st.sidebar:
    st.header("📊 Database Status")
    try:
        count = table.count_rows()
        st.success("✅ Connected")
        st.info(f"📄 {count:,} chunks indexed")
        with st.expander("Schema Details"):
            for fld in table.schema:
                st.text(f"• {fld.name}: {fld.type}")
    except Exception as e:
        st.error(f"❌ DB error: {e}")

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
if user_q := st.chat_input("Ask a question about the document"):
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.messages.append({"role":"user","content":user_q})

    # Search for context
    with st.status("🔍 Searching document...", expanded=False):
        df = search_docs(user_q, table, num_results=3)
        if not df.empty:
            st.write("📋 **Found relevant sections:**")
            for _, row in df.iterrows():
                text = row.get("text", "").strip()
                meta = row.get("metadata") or {}
                fname = meta.get("filename", "Unknown file")
                pages = meta.get("page_numbers", [])
                title = meta.get("title", "Untitled section")
                page_str = f"p. {', '.join(str(p) for p in pages)}" if pages else ""
                source = f"{fname} {page_str}".strip()

                st.markdown(
                    f"""
                    <div class="search-result">
                      <details>
                        <summary>📄 {source}</summary>
                        <div class="metadata">📖 Section: {title}</div>
                        <div class="content-text">{text}</div>
                      </details>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            context_text = build_context_string(df)
        else:
            st.warning("⚠️ No relevant content found.")
            context_text = ""

    # Get and display assistant response
    with st.chat_message("assistant"):
        assistant_reply = get_chat_response(st.session_state.messages, context_text)
    st.session_state.messages.append({"role":"assistant","content":assistant_reply})

    
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

# # def get_context(query: str, table, num_results: int = 5) -> str:
# #     """Get context from database search"""
# #     try:
# #         # Use the correct method - to_pandas() instead of to_df()
# #         results = table.search(query).limit(num_results).to_pandas()
# #         print(results)
        
# #         if results.empty:
# #             return "No context found - no search results returned."
        
# #         contexts = []

# #         for _, row in results.iterrows():
# #             try:
# #                 # Extract metadata safely with better error handling
# #                 metadata = row.get("metadata", {})
                
# #                 # Handle different metadata formats
# #                 if isinstance(metadata, str):
# #                     # If metadata is a string, try to parse it or use defaults
# #                     filename = "Unknown file"
# #                     page_numbers = []
# #                     title = "Untitled"
# #                 elif isinstance(metadata, dict):
# #                     filename = metadata.get("filename", "Unknown file")
# #                     page_numbers = metadata.get("page_numbers", [])
# #                     title = metadata.get("title", "Untitled")
# #                 else:
# #                     # If metadata is None or other type
# #                     filename = "Unknown file"
# #                     page_numbers = []
# #                     title = "Untitled"

# #                 # Safely handle text content
# #                 text_content = str(row.get('text', ''))
# #                 if not text_content.strip():
# #                     continue  # Skip empty text entries

# #                 # Build source citation safely
# #                 source_parts = []
# #                 if filename and filename != "Unknown file":
# #                     source_parts.append(str(filename))
                    
# #                 if page_numbers:
# #                     try:
# #                         if isinstance(page_numbers, list) and page_numbers:
# #                             page_str = ', '.join(str(p) for p in page_numbers if p is not None)
# #                             if page_str:
# #                                 source_parts.append(f"p. {page_str}")
# #                         elif page_numbers:
# #                             source_parts.append(f"p. {str(page_numbers)}")
# #                     except Exception as e:
# #                         print(f"Error processing page numbers: {e}")

# #                 source = f"\nSource: {' - '.join(source_parts) if source_parts else filename}"
                
# #                 if title and title != "Untitled":
# #                     source += f"\nTitle: {str(title)}"

# #                 # Add similarity score for debugging - handle safely
# #                 try:
# #                     distance = row.get('_distance', None)
# #                     if distance is not None and isinstance(distance, (int, float)):
# #                         similarity = 1 - float(distance)
# #                         source += f"\nSimilarity: {similarity:.3f}"
# #                     else:
# #                         source += f"\nSimilarity: N/A"
# #                 except Exception as e:
# #                     print(f"Error calculating similarity: {e}")
# #                     source += f"\nSimilarity: N/A"

# #                 contexts.append(f"{text_content}{source}")
                
# #             except Exception as row_error:
# #                 print(f"Error processing row: {row_error}")
# #                 # Continue processing other rows
# #                 continue

# #         if not contexts:
# #             return "No valid context found - all search results had processing errors."
            
# #         return "\n\n".join(contexts)
    
# #     except Exception as e:
# #         print(f"Search error details: {str(e)}")
# #         st.error(f"Error during search: {str(e)}")
# #         return "No context found due to search error."


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

# # Initialize Streamlit app
# st.set_page_config(
#     page_title="Document Q&A",
#     page_icon="📚",
#     layout="wide"
# )

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
#         with st.status("🔍 Searching documents...", expanded=False) as status:
#             # context = get_context(prompt, table)
#             context = table.search(query=prompt, query_type="vector").limit(5).to_df()
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

#             if context and context != "No context found due to search error.":
#                 st.write("📋 **Found relevant sections:**")
                
#                 for i, chunk in enumerate(context.split("\n\n")):
#                     if not chunk.strip():
#                         continue
                        
#                     # Split into text and metadata parts
#                     parts = chunk.split("\n")
#                     text = parts[0] if parts else ""
#                     metadata = {}
                    
#                     for line in parts[1:]:
#                         if ": " in line:
#                             key, value = line.split(": ", 1)
#                             metadata[key] = value

#                     source = metadata.get("Source", "Unknown source")
#                     title = metadata.get("Title", "Untitled section")
#                     similarity = metadata.get("Similarity", "N/A")

#                     st.markdown(
#                         f"""
#                         <div class="search-result">
#                             <details>
#                                 <summary>📄 {source} {f"(Similarity: {similarity})" if similarity != "N/A" else ""}</summary>
#                                 <div class="metadata">📖 Section: {title}</div>
#                                 <div class="content-text">{text}</div>
#                             </details>
#                         </div>
#                     """,
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
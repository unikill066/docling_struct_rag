# Document Q&A System

A Streamlit-based question-answering system that queries your LanceDB vector database of indexed documents.

## Features

- 🔍 **Vector Search**: Semantic search through your document database
- 💬 **Chat Interface**: Interactive Q&A with conversation history
- 📊 **Search Results**: Expandable view of relevant document sections
- 🎯 **Context-Aware**: GPT-4o-mini generates answers based on retrieved context
- 📈 **Similarity Scores**: Shows relevance scores for each search result

## Quick Start

### 1. Setup Environment

```bash
# Clone or download the files
# Navigate to the project directory

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy the template
cp .env.template .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Run the Application

```bash
# Option 1: Use the run script
chmod +x run_app.sh
./run_app.sh

# Option 2: Run directly
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Configuration

### Database Path
The default LanceDB path is set to:
```
/Users/discovery/Desktop/Docling/data/lancedb
```

To change this, either:
1. Update the path in `streamlit_app.py` in the `init_db()` function
2. Or set `LANCEDB_PATH` in your `.env` file

### Model Configuration
The app uses `gpt-4o-mini` by default. You can change this in the `get_chat_response()` function or add it to your `.env` file.

## Usage Tips

### Asking Good Questions
- **Be specific**: "What are the side effects of Drug X?" vs "Tell me about drugs"
- **Use domain terms**: Include technical terminology from your documents
- **Reference concepts**: Ask about specific methodologies, studies, or findings

### Understanding Results
- **Similarity scores**: Higher scores (closer to 1.0) indicate more relevant content
- **Source information**: Each result shows the document and page numbers
- **Expandable sections**: Click on results to see the full text

### Chat Features
- **Conversation history**: The app maintains context across questions
- **Follow-up questions**: Ask clarifying questions about previous answers
- **Clear context**: Start fresh by refreshing the page

## Troubleshooting

### Database Connection Issues
1. Verify the LanceDB path exists
2. Check that the table name "lance_neur_papers_db" is correct
3. Ensure you have read permissions for the database directory

### OpenAI API Issues
1. Verify your API key is correct in `.env`
2. Check your OpenAI account has sufficient credits
3. Ensure the model name is correct (default: gpt-4o-mini)

### Search Quality Issues
- **Too few results**: Lower the similarity threshold or increase num_results
- **Irrelevant results**: Try more specific query terms
- **No results**: Check if your query terms exist in the indexed documents

## File Structure

```
├── streamlit_app.py      # Main Streamlit application
├── requirements.txt      # Python dependencies
├── .env.template        # Environment variable template
├── .env                 # Your actual environment variables (not tracked)
├── run_app.sh           # Convenience script to start the app
├── README.md            # This file
└── misc/
    └── bmrn.jpeg        # Logo image (optional)
```

## Architecture

1. **Vector Search**: Uses LanceDB's vector search capabilities
2. **Context Building**: Combines relevant document chunks into context
3. **LLM Integration**: Feeds context to GPT-4o-mini for answer generation
4. **Streaming Response**: Real-time display of generated answers

## Customization

### Styling
Modify the CSS in the `st.markdown()` sections to change the appearance of search results and other UI elements.

### Search Parameters
Adjust these in the `search_docs()` function:
- `num_results`: Number of results to retrieve (default: 5)
- `query_type`: Type of search (default: "vector")

### Response Generation
Customize the system prompt in `get_chat_response()` to change how the AI responds to queries.

## Support

If you encounter issues:
1. Check the Streamlit logs in your terminal
2. Verify all environment variables are set correctly
3. Ensure your LanceDB is properly indexed and accessible





2025-08-06 21:30:52,856 - docling.document_converter - INFO - Finished converting document 2025.03.26.645611v1.full.pdf in 87.48 sec.
2025-08-06 21:30:54,391 - doc_pipeline - INFO - Converted 7 documents.
2025-08-06 21:30:54,939 - doc_pipeline - INFO - Chunked document into 30 chunks.
2025-08-06 21:30:55,091 - doc_pipeline - INFO - Chunked document into 16 chunks.
2025-08-06 21:30:55,309 - doc_pipeline - INFO - Chunked document into 29 chunks.
2025-08-06 21:30:55,572 - doc_pipeline - INFO - Chunked document into 17 chunks.
2025-08-06 21:30:56,012 - doc_pipeline - INFO - Chunked document into 23 chunks.
2025-08-06 21:30:56,274 - doc_pipeline - INFO - Chunked document into 26 chunks.
2025-08-06 21:30:56,769 - doc_pipeline - INFO - Chunked document into 36 chunks.
2025-08-06 21:30:56,769 - doc_pipeline - INFO - Preparing chunks for embedding...
2025-08-06 21:30:56,770 - doc_pipeline - INFO - Storing chunks in LanceDB...
[2025-08-07T02:30:56Z WARN  lance::dataset::write::insert] No existing dataset at /Users/discovery/Desktop/Docling/data/lancedb/lance_neur_papers_db.lance, it will be created
2025-08-06 21:31:00,257 - httpx - INFO - HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
2025-08-06 21:31:00,807 - doc_pipeline - INFO - Stored 177 rows.
2025-08-06 21:31:00,839 - doc_pipeline - INFO -                                                   text                                             vector                                           metadata
0    ´ Urzula Franco-Enz ´ astiga a , Nikhil N. Int...  [0.013282559, -0.011569558, 0.0014011605, 0.01...  {'filename': 'epigenomic_landscape_of_the_huma...
1    Cell states are influenced by the regulation o...  [0.038289737, 0.021240013, 0.060771782, 0.0477...  {'filename': 'epigenomic_landscape_of_the_huma...
2    Women experience greater sensitivity to nocice...  [0.021032156, 0.0036895177, 0.037771516, 0.046...  {'filename': 'epigenomic_landscape_of_the_huma...
3    All human tissue procurement procedures were a...  [0.020772401, -0.02706411, 0.03680161, 0.06093...  {'filename': 'epigenomic_landscape_of_the_huma...
4    Immediately after dissection, human L4 or L5 D...  [0.029853573, 0.002353555, 0.022892665, 0.0145...  {'filename': 'epigenomic_landscape_of_the_huma...
..                                                 ...                                                ...                                                ...
172  Our work is an example of the importance of a ...  [0.023956355, 0.015335159, 0.005254547, 0.0355...  {'filename': '2025.03.26.645611v1.full.pdf', '...
173  1. N. B. Finnerup, N. Attal, S. Haroutounian, ...  [-0.0015025215, 0.01183183, 0.008757842, 0.069...  {'filename': '2025.03.26.645611v1.full.pdf', '...
174  71. A. Bavencoffe, E. A. Spence, M. Y. Zhu, A....  [0.0017224993, 0.00889958, 0.0066244453, 0.062...  {'filename': '2025.03.26.645611v1.full.pdf', '...
175  OSMR*IGFRA2\nOSMR IGFRA2*\nOSMR IGFRA2\nOSMR F...  [0.02049529, 0.014894433, 0.004708563, 0.01249...  {'filename': '2025.03.26.645611v1.full.pdf', '...
176  Fig. 2. Immunofluorescence labeling of human p...  [-0.0022837063, 0.016030857, 0.016017038, 0.02...  {'filename': '2025.03.26.645611v1.full.pdf', '...

[177 rows x 3 columns]
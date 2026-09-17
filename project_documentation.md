# PaperMindAI - Project Architecture & Documentation

## Overview
**PaperMindAI** is a Retrieval-Augmented Generation (RAG) system built to allow users to interact intelligently with complex PDF research papers. By combining local vector search with advanced Large Language Models (LLMs), it can accurately answer user queries based *strictly* on the uploaded documents, drastically reducing hallucinations. It also features a transparent citation system that allows users to verify answers against the original PDF pages.

---

## 🛠️ The Technology Stack and Justifications

1. **Frontend & UI: Streamlit**
   - **Why**: Streamlit is the industry standard for rapidly prototyping Python-based AI applications. It allows us to build a responsive, interactive chat interface with file upload capabilities, session state management, and sidebar configurations without writing any HTML/React code.

2. **Orchestration: LangChain**
   - **Why**: LangChain provides robust, pre-built abstractions for the entire RAG pipeline. We utilize its Document objects, TextSplitters, VectorStore integrations, and LangChain Expression Language (LCEL) chains to stitch together retrieval and generation seamlessly.

3. **PDF Parsing: PyMuPDF (`pymupdf4llm` & `fitz`)**
   - **Why**: Traditional PDF loaders (like PyPDF2) often mangle tables and multi-column layouts found in academic papers. `pymupdf4llm` extracts the PDF natively into structured **Markdown**. This ensures tables and code blocks remain structurally intact, making the LLM much better at reading them. 
   - `fitz` is used to natively extract images and graphs from the PDF to be passed to a Vision model for summarization.

4. **Embeddings:**
   - **HuggingFace (`BAAI/bge-small-en`)**: A highly optimized open-source embedding model that runs completely locally on the CPU. This guarantees privacy and costs $0 in API fees, while maintaining state-of-the-art semantic search accuracy.
   - **Google Gemini (`gemini-embedding-001`)**: Provided as a commercial fallback for slightly deeper semantic understanding.

5. **Vector Database: FAISS (Facebook AI Similarity Search)**
   - **Why**: FAISS is a lightweight, strictly local, in-memory vector database. For a desktop/local web app like this, setting up a cloud database (like Pinecone) or a heavy local service (like pgvector) is overkill. FAISS provides blazing-fast nearest-neighbor search with zero setup.

6. **Large Language Model: Google Gemini (3.1 Flash / Lite)**
   - **Why**: Gemini Flash models are optimized for incredibly low latency and very large context windows. Because we are passing thousands of words of retrieved context in every prompt, a fast model is required to ensure the user isn't waiting 30 seconds for a response.

---

## ⚙️ How It Works (Top-to-Bottom Workflow)

### 1. Document Upload & Processing (`app.py` & `loader.py`)
- The user uploads one or multiple PDF documents via the Streamlit sidebar.
- `loader.py` opens the PDF and uses `pymupdf4llm` to extract every page as structured Markdown text.
- If **Multimodal Extraction** is enabled, it searches every page for images/graphs, extracts the raw image bytes, and sends them to Gemini Vision to generate a text summary of the graph.
- These text chunks and image summaries are converted into LangChain `Document` objects containing metadata (Filename and Page Number).

### 2. Text Chunking (`chunking.py`)
- You cannot feed a 100-page PDF entirely into an embedding model; it must be broken down.
- The `MarkdownTextSplitter` divides the documents into chunks (e.g., 1000 characters with a 200-character overlap to prevent cutting sentences in half). 
- Because we use a *Markdown* splitter, it intelligently avoids breaking chunks in the middle of a Markdown table or a bulleted list.

### 3. Embedding and Indexing (`embeddings.py` & `vectorstore.py`)
- Each chunk is passed through the chosen Embedding Model (HuggingFace or Gemini), transforming the text into an array of floating-point numbers (vectors) that represent the semantic meaning of the text.
- `vectorstore.py` takes these vectors and creates a **FAISS Index**. 
- It also handles rate-limiting. If the user selects Gemini embeddings (which have strict limits on free tiers), the script processes the chunks in small batches and automatically sleeps/retries if a `429 Too Many Requests` error occurs. The FAISS index is then saved locally to the disk (`faiss_index/`).

### 4. Querying & Retrieval (`retriever.py` & `app.py`)
- When the user types a question, the system converts that question into a vector using the exact same embedding model.
- It queries the FAISS index for the vectors most mathematically similar to the query vector. 
- It uses either pure **Similarity Search** or **MMR (Maximal Marginal Relevance)**. MMR fetches a larger pool of similar documents and then filters them to ensure the final context is diverse, rather than pulling 3 identical paragraphs.

### 5. Answer Generation (`rag_pipeline.py`)
- The top chunks (with their respective page numbers) are joined into one large "Context" string.
- This context, along with the user's query, is injected into a strict system prompt.
- **The Prompt Rules**: The system is specifically instructed to *only* use the provided context. If the answer isn't in the context, it is hardcoded to reply EXACTLY with `"I don't know"`. 
- The LLM processes the prompt and streams back the answer.

### 6. Citation and Verification (`app.py`)
- The response is displayed to the user.
- The UI takes the metadata from the chunks used to generate the answer and creates dropdown expanders under **"Citing Sources"**.
- Inside these expanders, users can read the exact raw text chunk used. Furthermore, through a dynamic toggle, `app.py` can re-open the original PDF file and render a live image of the exact page the data was pulled from, ensuring 100% verifiability and trust.

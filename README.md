# 🤖 PaperMindAI

PaperMindAI is a local **Retrieval-Augmented Generation (RAG)** agent designed to help you intelligently interact and chat with complex PDF research papers. By combining local vector search with advanced Large Language Models (LLMs), it answers your queries based *strictly* on the uploaded documents, drastically reducing hallucinations.

## ✨ Features
*   **Strict Context Adherence:** The LLM is hardcoded to answer *only* using the provided documents. If the answer isn't there, it responds with "I don't know."
*   **Live Verifiable Citations:** Click on "Citing Sources" in the UI to see the exact text chunk used. You can toggle a live image rendering of the original PDF page to verify the data.
*   **Intelligent Markdown Extraction:** Uses `pymupdf4llm` to extract PDFs directly into structured Markdown, perfectly preserving tables, lists, and code blocks.
*   **Multimodal Image Extraction (Optional):** Automatically finds graphs/charts in the PDF and uses Vision AI to summarize them into text, allowing you to ask questions about visual data.
*   **Local Privacy Options:** Configure the pipeline to use local HuggingFace embeddings (`BAAI/bge-small-en`) for complete privacy and zero API costs.
*   **Advanced Retrieval Strategies:** Choose between standard Similarity Search or Maximal Marginal Relevance (MMR) to ensure diverse context retrieval.

## 🗂️ Project Structure
*   `app.py`: Main Streamlit application and UI components.
*   `loader.py`: Handles PDF parsing and multimodal extraction using PyMuPDF.
*   `chunking.py`: Splits documents using intelligent Markdown text splitters.
*   `embeddings.py`: Configures HuggingFace and Gemini embedding models.
*   `vectorstore.py`: Creates, loads, and manages the local FAISS index.
*   `retriever.py`: Manages Similarity and MMR retrieval strategies.
*   `rag_pipeline.py`: Defines the LangChain expression language (LCEL) chain for answer generation.

## 🏗️ Architecture
1.  **Document Upload & Processing:** Uploaded PDFs are parsed via PyMuPDF into structured Markdown. Multimodal extraction pulls and summarizes images/graphs.
2.  **Text Chunking:** Markdown splitters divide documents into smaller chunks (e.g., 1000 or 1500 characters) while preserving table and list structures.
3.  **Embedding and Indexing:** Chunks are transformed into vectors using HuggingFace or Gemini embeddings and stored in a local FAISS index.
4.  **Querying & Retrieval:** User queries are embedded and matched against the FAISS index using Similarity or MMR search to retrieve relevant chunks.
5.  **Answer Generation:** The retrieved context is injected into a strict system prompt. The LLM generates an answer, strictly abiding by the provided context.
6.  **Citation and Verification:** The response is displayed alongside verifiable citations and original PDF page snapshots.

## ⚙️ Tech Stack
*   **Frontend UI:** Streamlit
*   **Orchestration:** LangChain (LCEL)
*   **PDF Parsing:** PyMuPDF (`pymupdf4llm` & `fitz`)
*   **Vector Database:** FAISS
*   **Embeddings:** HuggingFace (`bge-small-en`) or Google Gemini (`gemini-embedding-001`)
*   **LLM:** Google Gemini 3.1 Flash / Lite

## ⚙️ Prerequisites
*   Python 3.8+
*   Google Gemini API Key (required for LLM inference and Gemini embeddings)
*   A virtual environment is highly recommended.

## 🧪 Embedding Comparison
*   **HuggingFace (`BAAI/bge-small-en`)**: A completely local baseline. Incurs zero API cost, runs entirely offline, and performs flawlessly on basic semantic proximity.
*   **Google Gemini**: Commercial endpoint that yields noticeably stronger performance at capturing abstract context and dense logic across multi-hop scenarios.

## 🔍 Retrieval Strategy Comparison
*   **Similarity Search**: Focuses strictly on cosine proximity. Best for direct empirical lookups, definitions, or exact constants.
*   **MMR (Maximal Marginal Relevance)**: Penalizes chunks that are too similar to already selected ones, increasing semantic diversity. Best for complex synthesis, multi-hop queries, and heavily technical context.

## 📊 Testing & Evaluation
The pipeline was extensively evaluated across multiple queries:
*   **Simple Fact Retrieval:** Strategy A (1000/200) + Similarity Search performed exceptionally well.
*   **Complex Concept Synthesis:** Strategy B (1500/300) + MMR significantly outperformed Similarity Search.
*   **Missing Information Check:** Validated the strict system prompt; the pipeline dependably responded with "I don't know" when information was absent.
*   **Quantitative/Formulas:** PyMuPDF4LLM extraction successfully preserved markdown formatting for equations and tables.

## 🧠 Key Feature (Strict QA Mode)
The system is governed by a strict prompt directive. The LLM is forced to rely **only** on the context retrieved from the FAISS database. If a user asks a question outside the scope of the uploaded documents (e.g., "Who won the World Cup in 2022?"), the model will refuse to hallucinate and instead reply with, "I don't know".

## 🚀 Setup Instructions

### 1. Clone or Download the Project
```bash
git clone https://github.com/Lochanarajesh/PaperMindAI.git
cd PaperMindAI
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key
Create a `.env` file in the root directory and add your Google Gemini API Key:
```env
GEMINI_API_KEY=your_api_key_here
```

### 5. Run the App
```bash
python -m streamlit run app.py
```

## 📸 Demo Screenshot
![PaperMindAI Demo](demo_screenshot.png) *(Add a screenshot here)*

## 📖 How to Use
1.  Launch the app and open the `localhost` URL in your browser.
2.  Open the sidebar to configure your Embedding Model, Chunking Strategy, and Retrieval Method.
3.  Upload one or multiple PDF research papers and wait for the processing to finish.
4.  Type your queries in the chat box.
5.  Expand the citations under the answers to read the raw text or view the original PDF page snippet to verify the information.

## 🛠️ Troubleshooting
*   **Rate Limits:** If you encounter a `429 Too Many Requests` error with Gemini embeddings, wait a few moments or switch to HuggingFace embeddings in the sidebar.
*   **Stale Data:** If the app behaves unexpectedly, click the **Reset** button to clear the FAISS vectorstore and session state.
*   **Images Not Loading:** Ensure you enable the "Enable Image Extraction" checkbox before uploading documents if you want multimodal capabilities.

## 📦 Dependencies
*   `streamlit`
*   `langchain`
*   `pymupdf4llm`
*   `pymupdf` (fitz)
*   `faiss-cpu`
*   `python-dotenv`
*   `pillow`

## 🚧 Future Improvements (Optional Enhancements)
*   Support for multiple vector databases (e.g., Chroma, Qdrant).
*   Integration with open-source local LLMs (e.g., Llama 3) via Ollama for a fully offline pipeline.
*   Advanced reranking capabilities (e.g., Cohere Rerank) to further boost retrieval precision.

## 🔒 Privacy & Security
If you select the HuggingFace (`BAAI/bge-small-en`) embedding option, your document data is embedded **100% locally**. The vector database (FAISS) is entirely local and stores indices on your machine. Data is only transmitted externally when generating responses via the Gemini API.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
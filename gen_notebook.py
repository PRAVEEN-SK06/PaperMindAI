import json
import os

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📄 PaperMindAI: Capstone Project\n",
            "**Subtitle**: Local Retrieval-Augmented Generation (RAG) System for Complex PDF Research Papers\n\n",
            "This notebook documents the fully implemented, end-to-end architecture and evaluation of the PaperMindAI project. It runs the entire pipeline from document ingestion to the final generated response with citations."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Problem Statement\n",
            "**Business Context**: Researchers, students, and professionals spend countless hours manually reading and parsing lengthy PDF documents to find specific information or answer nuanced questions.\n",
            "**Objective**: Build an automated, hallucination-free Question Answering (QA) system that accurately retrieves information directly from uploaded documents.\n",
            "**Use Case**: A system where users upload research papers, ask natural language questions, and receive precise answers backed by inline citations and original source page images."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Setup & Imports\n",
            "We import the custom modules built for this project: `loader`, `chunking`, `embeddings`, `vectorstore`, `retriever`, and `rag_pipeline`."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "from dotenv import load_dotenv\n\n",
            "# Load environment variables (like GEMINI_API_KEY)\n",
            "load_dotenv()\n",
            "api_key = os.getenv(\"GEMINI_API_KEY\", \"\")\n\n",
            "# Import custom project modules\n",
            "from loader import load_documents_from_paths\n",
            "from chunking import split_documents\n",
            "from embeddings import get_huggingface_embeddings, get_gemini_embeddings\n",
            "from vectorstore import create_and_save_vectorstore\n",
            "from retriever import get_retriever\n",
            "from rag_pipeline import generate_answer"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Data Ingestion\n",
            "**Dataset Details**: Any unstructured PDF files. For this experiment, we use an academic research paper located in `pdf_uploads/`.\n",
            "**Loader Used**: PyMuPDF (`fitz` and `pymupdf4llm`). We chose this because standard PyPDF often mangles tables, whereas PyMuPDF natively extracts pages into structured Markdown and supports image extraction for multimodal RAG."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load a sample PDF from our uploads directory\n",
            "sample_pdf = \"pdf_uploads/RAG.pdf\"\n\n",
            "print(f\"Loading document: {sample_pdf}\")\n",
            "docs = load_documents_from_paths([sample_pdf], api_key=api_key, extract_images=False)\n",
            "print(f\"Loaded {len(docs)} pages.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Text Chunking\n",
            "**Chunking Strategies Compared**:\n",
            "- **Strategy A**: Chunk Size 1000, Overlap 200. Good for general dense text and definitions.\n",
            "- **Strategy B**: Chunk Size 1500, Overlap 300. Good for complex academic papers, tables, and code.\n\n",
            "**Chosen Strategy**: We use **Strategy B** because the larger chunk size keeps complex code elements, tables, and long-form paragraphs together, preventing loss of context. The 300-character overlap ensures that critical sentences spanning across chunks are not broken in half."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Split the loaded document into manageable chunks\n",
            "chunks = split_documents(docs, strategy=\"B\")\n",
            "print(f\"Document split into {len(chunks)} chunks.\")\n",
            "print(f\"Sample chunk preview: {chunks[0].page_content[:150]}...\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Embeddings\n",
            "**Models Compared**:\n",
            "- **Open Source**: HuggingFace (`BAAI/bge-small-en`). Fast, runs entirely locally, ensures data privacy, zero API costs.\n",
            "- **Commercial**: Google Gemini (`gemini-embedding-001`). Offers state-of-the-art semantic understanding but requires API access.\n\n",
            "**Final Choice**: We support both, but for this notebook, we use the local **HuggingFace** embeddings to demonstrate a fast, privacy-preserving semantic representation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Initialize the HuggingFace embedding model\n",
            "print(\"Initializing HuggingFace Embeddings...\")\n",
            "embeddings = get_huggingface_embeddings()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Vector DB & Retrieval\n",
            "**Vector Database Used**: **FAISS** (Facebook AI Similarity Search). Chosen for its blazing fast, in-memory execution, perfect for local setups.\n\n",
            "**Retrieval Strategies Compared**:\n",
            "- **Similarity**: Standard Cosine Similarity. Quickest for exact snippets.\n",
            "- **MMR (Max Marginal Relevance)**: Relevance + Diversity. Best for summaries spanning multiple sections.\n",
            "- **Hybrid Search**: Combines BM25 (keyword search) with FAISS (semantic search).\n",
            "- **Reranker**: Uses an MMR base retriever wrapped in a HuggingFace Cross-Encoder to re-score chunks.\n\n",
            "**Final Selection**: We will demonstrate the **Reranker** strategy to maximize the precision of the final context window."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create the FAISS Vector Store\n",
            "print(\"Building FAISS Vector Store...\")\n",
            "vectorstore = create_and_save_vectorstore(chunks, embeddings)\n\n",
            "# Configure the Retriever\n",
            "# Using the advanced 'reranker' strategy (MMR + Cross-Encoder) and fetching top 3 chunks\n",
            "retriever = get_retriever(vectorstore, strategy=\"reranker\", top_k=3)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. RAG Pipeline\n",
            "**Chain Construction**: Built using LangChain Expression Language (LCEL).\n",
            "**Prompt Template**: Extremely strict instructions mandating the LLM to *only* use provided context, output \"I don't know\" if the answer isn't present (zero hallucinations), and highlight \"Note:\" segments.\n",
            "**LLM Used**: Google Gemini (`gemini-3.1-flash-lite-preview`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define a test query\n",
            "query = \"What is the main contribution or objective discussed in this paper?\"\n\n",
            "print(f\"Querying the LLM: '{query}'\\n\")\n\n",
            "# Run the generation pipeline\n",
            "answer, retrieved_docs = generate_answer(\n",
            "    query=query,\n",
            "    retriever=retriever,\n",
            "    api_key=api_key,\n",
            "    model_name=\"gemini-3.1-flash-lite-preview\",\n",
            "    chat_history=[]\n",
            ")\n\n",
            "print(\"### GENERATED ANSWER ###\")\n",
            "print(answer)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Results & Source Citations\n",
            "The final step is to verify the results. We ensure the LLM returns exactly the **Top 3 supporting citations**, including the exact paper title/filename and page number."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Display the top 3 citations\n",
            "print(\"### TOP 3 SUPPORTING CITATIONS ###\\n\")\n\n",
            "for idx, doc in enumerate(retrieved_docs[:3]):\n",
            "    # Extract metadata safely\n",
            "    source = doc.metadata.get('source', 'Unknown Document')\n",
            "    filename = os.path.basename(source) if source != 'Unknown Document' else 'Unknown Document'\n",
            "    page = doc.metadata.get('page', 'Unknown Page')\n",
            "    \n",
            "    print(f\"Citation [{idx + 1}]:\")\n",
            "    print(f\"- Paper Title / File: {filename}\")\n",
            "    print(f\"- Page Number: {page}\")\n",
            "    print(f\"- Snippet Preview: {doc.page_content[:200].replace(chr(10), ' ')}...\\n\")\n"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('PaperMindAI_Capstone.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

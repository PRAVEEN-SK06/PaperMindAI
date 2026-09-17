import os
from dotenv import load_dotenv

# Load environment variables (like GEMINI_API_KEY)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY", "")

# Import custom project modules
from loader import load_documents_from_paths
from chunking import split_documents
from embeddings import get_huggingface_embeddings, get_gemini_embeddings
from vectorstore import create_and_save_vectorstore
from retriever import get_retriever
from rag_pipeline import generate_answer
# Load a sample PDF from our uploads directory
sample_pdf = "pdf_uploads/2410.15944v1.pdf"

print(f"Loading document: {sample_pdf}")
docs = load_documents_from_paths([sample_pdf], api_key=api_key, extract_images=False)
print(f"Loaded {len(docs)} pages.")
# Split the loaded document into manageable chunks
chunks = split_documents(docs, strategy="B")
print(f"Document split into {len(chunks)} chunks.")
print(f"Sample chunk preview: {chunks[0].page_content[:150]}...")
# Initialize the HuggingFace embedding model
print("Initializing HuggingFace Embeddings...")
embeddings = get_huggingface_embeddings()
# Create the FAISS Vector Store
print("Building FAISS Vector Store...")
vectorstore = create_and_save_vectorstore(chunks, embeddings)

# Configure the Retriever
# Using the advanced 'reranker' strategy (MMR + Cross-Encoder) and fetching top 3 chunks
retriever = get_retriever(vectorstore, strategy="reranker", top_k=3)
# Define a test query
query = "What is the main contribution or objective discussed in this paper?"

print(f"Querying the LLM: '{query}'\n")

# Run the generation pipeline
answer, retrieved_docs = generate_answer(
    query=query,
    retriever=retriever,
    api_key=api_key,
    model_name="gemini-3.1-flash-lite-preview",
    chat_history=[]
)

print("### GENERATED ANSWER ###")
print(answer)
# Display the top 3 citations
print("### TOP 3 SUPPORTING CITATIONS ###\n")

for idx, doc in enumerate(retrieved_docs[:3]):
    # Extract metadata safely
    source = doc.metadata.get('source', 'Unknown Document')
    filename = os.path.basename(source) if source != 'Unknown Document' else 'Unknown Document'
    page = doc.metadata.get('page', 'Unknown Page')
    
    print(f"Citation [{idx + 1}]:")
    print(f"- Paper Title / File: {filename}")
    print(f"- Page Number: {page}")
    print(f"- Snippet Preview: {doc.page_content[:200].replace(chr(10), ' ')}...\n")

import os
import shutil
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

FAISS_INDEX_PATH = "faiss_index"

def clear_vectorstore(index_path: str = FAISS_INDEX_PATH):
    """
    Deletes the local FAISS vector database directory.
    """
    if os.path.exists(index_path):
        try:
            shutil.rmtree(index_path)
            print(f"Deleted FAISS index at {index_path}")
        except Exception as e:
            print(f"Error deleting FAISS index: {e}")


import time

def create_and_save_vectorstore(chunks: List[Document], embeddings: Embeddings, index_path: str = FAISS_INDEX_PATH) -> FAISS:
    """
    Creates a FAISS vector database from document chunks and saves it locally.
    Includes batching and rate-limit handling for free-tier APIs.
    """
    if not chunks:
        raise ValueError("No chunks provided to create the vector store.")
        
    print(f"Creating FAISS index with {len(chunks)} chunks...")
    
    # Process in batches to avoid rate limits (429 errors)
    batch_size = 10
    vectorstore = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}...")
        
        # Backoff loop for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break # Success, exit retry loop
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        sleep_time = 22 # Wait longer than the retry delay suggested in the error
                        print(f"Rate limit hit. Waiting {sleep_time} seconds before retrying...")
                        time.sleep(sleep_time)
                    else:
                        raise e
                else:
                    raise e
                    
        # Small delay between successful batches to pace the requests
        if i + batch_size < len(chunks):
            time.sleep(2)
    
    # Save locally
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)
    print(f"FAISS index saved to {index_path}")
    
    return vectorstore

def load_vectorstore(embeddings: Embeddings, index_path: str = FAISS_INDEX_PATH) -> Optional[FAISS]:
    """
    Loads a locally saved FAISS vector database.
    Returns None if the directory doesn't exist.
    """
    if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
        print(f"Loading FAISS index from {index_path}...")
        # allow_dangerous_deserialization=True is needed since LangChain 0.2.x 
        # for loading locally trusted FAISS indexes.
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    else:
        print(f"No FAISS index found at {index_path}")
        return None

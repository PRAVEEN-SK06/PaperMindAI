from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

def split_documents(documents: List[Document], strategy: str = "A") -> List[Document]:
    """
    Split a list of documents into chunks using MarkdownTextSplitter.
    Conserves metadata (source and page) automatically over the chunks.
    
    Strategies:
      "A": chunk_size = 1000, chunk_overlap = 200
      "B": chunk_size = 1500, chunk_overlap = 300
    """
    if strategy == "A":
        chunk_size = 1000
        chunk_overlap = 200
    elif strategy == "B":
        chunk_size = 1500
        chunk_overlap = 300
    else:
        # Default fallback
        chunk_size = 1000
        chunk_overlap = 200

    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # split_documents method automatically duplicates the metadata for each chunk
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split {len(documents)} documents into {len(chunks)} chunks using Strategy {strategy} ({chunk_size}/{chunk_overlap}).")
    return chunks

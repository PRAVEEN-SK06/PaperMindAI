import os
import warnings

# Suppress warnings (e.g., Pydantic, tokenizers, USER_AGENT)
warnings.filterwarnings("ignore")
os.environ["USER_AGENT"] = "PaperMindAI"

# Disable Xet optimized storage to avoid OSError on model download
os.environ["HF_HUB_DISABLE_XET"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_huggingface_embeddings(model_name: str = "BAAI/bge-small-en") -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFace embedding model.
    Default uses BAAI/bge-small-en as requested.
    """
    # model_kwargs = {'device': 'cpu'} can be added if GPU is not available, but transformers handles it
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        # Set encode_kwargs to normalize embeddings for cosine similarity
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return embeddings

def get_gemini_embeddings(api_key: str, model_name: str = "models/gemini-embedding-001") -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Gemini embedding model (Commercial).
    """
    if not api_key:
        raise ValueError("API Key is required for Gemini Embeddings.")
        
    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key
    )
    return embeddings

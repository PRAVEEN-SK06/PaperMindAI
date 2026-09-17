import os
import base64
import fitz  # PyMuPDF
import pymupdf4llm
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

def summarize_image(img_bytes: bytes, api_key: str) -> str:
    """
    Passes an image to Gemini Vision to extract a description and any tabular/text data.
    """
    if not api_key:
        return ""
        
    try:
        encoded_image = base64.b64encode(img_bytes).decode('utf-8')
        
        # Use gemini-3.1-flash-latest for fast multimodal extraction
        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-latest", google_api_key=api_key)
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": "Describe this image in detail. Focus heavily on extracting any tables, data, flowcharts, or text present in the image. Format tables clearly."
                },
                {
                    "type": "image_url", 
                    "image_url": f"data:image/jpeg;base64,{encoded_image}"
                }
            ]
        )
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error summarizing image: {e}")
        return ""

def load_documents_from_paths(file_paths: List[str], api_key: str = "", extract_images: bool = False) -> List[Document]:
    """
    Load PDFs from a list of file paths.
    Uses pymupdf4llm to extract structured markdown (perfect for tables) 
    and optionally uses fitz to extract images for Gemini summarization.
    """
    all_docs = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        try:
            basename = os.path.basename(file_path)
            
            # 1. Extract Markdown Text structured by pages
            md_pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
            for page_data in md_pages:
                text = page_data.get("text", "")
                page_num = page_data.get("metadata", {}).get("page_number", 1)
                if text.strip():
                    all_docs.append(Document(
                        page_content=text,
                        metadata={"source": basename, "page": page_num, "type": "text"}
                    ))
            
            # 2. Extract Images using raw PyMuPDF
            if extract_images:
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        print(f"Extracting image {img_index+1} from {basename} page {page_num+1}...")
                        summary = summarize_image(image_bytes, api_key)
                        
                        if summary:
                            image_content = f"--- Image/Visual Data (Page {page_num + 1}) ---\n{summary}"
                            all_docs.append(Document(
                                page_content=image_content,
                                metadata={"source": basename, "page": page_num + 1, "type": "image"}
                            ))
                doc.close()
                print(f"Loaded {len(md_pages)} pages with Multimodal extraction from {file_path}")
            else:
                print(f"Loaded {len(md_pages)} pages (text only) from {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return all_docs

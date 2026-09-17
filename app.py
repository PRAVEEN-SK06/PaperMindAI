import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
import fitz
from PIL import Image
from dotenv import load_dotenv

# ------------------ PDF IMAGE ------------------
@st.cache_data
def get_pdf_page_image(pdf_path, page_num):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(max(0, int(page_num) - 1))
        pix = page.get_pixmap(dpi=150)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except:
        return None

# ------------------ IMPORTS ------------------
from loader import load_documents_from_paths
from chunking import split_documents
from embeddings import get_huggingface_embeddings, get_gemini_embeddings
from vectorstore import create_and_save_vectorstore, clear_vectorstore, load_vectorstore
from retriever import get_retriever
from rag_pipeline import generate_answer

load_dotenv()

# ------------------ CONFIG ------------------
st.set_page_config(page_title="PaperMindAI", page_icon="🤖", layout="wide")

# ------------------ CSS ------------------
st.markdown("""
<style>

.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    height: 45px;
    font-size: 15px;
    font-weight: 600;
    border: none;
    color: white !important;
}

.stTextInput>div>div>input {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ SESSION ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.markdown("## ⚙️ Configurations")
    st.markdown("---")

    # Document Processing
    st.markdown("### 📄 Document Processing")

    embedding_strategy = st.radio(
        "Embedding Model",
        [
            "HuggingFace (BAAI/bge-small-en)",
            "Google Gemini (gemini-embedding-001)"
        ]
    )

    chunking_strategy = st.radio(
        "Chunking Strategy",
        [
            "A (1000 size / 200 overlap)",
            "B (1500 size / 300 overlap)"
        ]
    )

    st.markdown("---")

    # Retrieval
    st.markdown("### 🔍 Retrieval Settings")

    retrieval_strategy = st.radio(
        "Retrieval Strategy",
        [
            "Similarity",
            "MMR",
            "Hybrid",
            "Reranker"
        ]
    )

# ------------------ TITLE ------------------
st.title("🤖 PaperMindAI")

# ------------------ FILE UPLOAD ------------------
enable_multimodal = st.checkbox("Enable Image Extraction (Slower)", value=False)
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

col1, col2 = st.columns(2)

# ------------------ PROCESS ------------------
with col1:
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):

                upload_dir = "pdf_uploads"
                os.makedirs(upload_dir, exist_ok=True)

                file_paths = []
                for file in uploaded_files:
                    path = os.path.join(upload_dir, file.name)
                    with open(path, "wb") as f:
                        f.write(file.getvalue())
                    file_paths.append(path)

                strategy_map = {
                    "A (1000 size / 200 overlap)": "A",
                    "B (1500 size / 300 overlap)": "B"
                }

                docs = load_documents_from_paths(
                    file_paths,
                    api_key=os.getenv("GEMINI_API_KEY", ""),
                    extract_images=enable_multimodal
                )

                chunks = split_documents(
                    docs,
                    strategy=strategy_map.get(chunking_strategy, "A")
                )

                if "HuggingFace" in embedding_strategy:
                    embeddings = get_huggingface_embeddings()
                else:
                    embeddings = get_gemini_embeddings(api_key=os.getenv("GEMINI_API_KEY", ""))

                vs = create_and_save_vectorstore(chunks, embeddings)

                st.session_state.vectorstore = vs
                st.success("Documents Processed!")

# ------------------ RESET ------------------
with col2:
    if st.button("Reset"):
        clear_vectorstore()
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.rerun()

# ------------------ CHAT DISPLAY ------------------
for msg_idx, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if "citations" in msg and msg["citations"]:
            st.markdown("##### 📚 Reference Documents")
            for idx, doc in enumerate(msg["citations"]):
                import re
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                content = re.sub(r'\n{3,}', '\n\n', doc.page_content).strip()
                with st.expander(f"📄 [{idx+1}] {source} (Page {page})"):
                    st.markdown(content)
                    if st.toggle("📸 View Original Page", key=f"history_toggle_{msg_idx}_{idx}_{page}_{source}"):
                        img = get_pdf_page_image(source, page)
                        if img:
                            st.image(img, caption=f"{source} - Page {page}", use_container_width=True)
                        else:
                            st.error("Document is no longer available on disk.")

# ------------------ CHAT INPUT ------------------
prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Load vectorstore if missing
    if st.session_state.vectorstore is None:
        try:
            if "HuggingFace" in embedding_strategy:
                embeddings = get_huggingface_embeddings()
            else:
                embeddings = get_gemini_embeddings(api_key=os.getenv("GEMINI_API_KEY", ""))
            vs = load_vectorstore(embeddings)
            if vs:
                st.session_state.vectorstore = vs
        except:
            pass

    if st.session_state.vectorstore:
        with st.spinner("Thinking..."):

            strategy_map = {
                "Similarity": "similarity",
                "MMR": "mmr",
                "Hybrid": "hybrid",
                "Reranker": "reranker"
            }

            retriever = get_retriever(
                st.session_state.vectorstore,
                strategy=strategy_map.get(retrieval_strategy, "similarity"),
                top_k=3
            )

            answer, docs = generate_answer(
                query=prompt,
                retriever=retriever,
                api_key=os.getenv("GEMINI_API_KEY", ""),
                model_name=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
                chat_history=st.session_state.chat_history
            )

            # Safety fallback
            if not docs:
                answer = "I don't know"

            # Avoid showing references if the model doesn't know the answer
            if "i don't know" in answer.lower().strip() or "i dont't know" in answer.lower().strip():
                docs = []

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": docs
            })

            st.rerun()
    else:
        st.warning("Upload documents first")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Powered by PaperMindAI")
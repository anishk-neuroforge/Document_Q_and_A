import os
import time
import pathlib
import streamlit as st
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from utils.prompts import SYSTEM_PROMPT

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

BASE_DIR   = pathlib.Path(__file__).parent
UPLOAD_DIR = str(BASE_DIR / "uploads")
VECTOR_DIR = str(BASE_DIR / "vectorstore")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

MAX_CONTEXT_CHARS = 3000

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Document Q&A — RAG Pipeline",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Q&A — RAG Pipeline")
st.markdown("Semantic Retrieval + Grounded Source Citations over PDF documents")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("⚙️ System Info")
st.sidebar.markdown("""
### Active Stack
- LangChain
- FAISS
- all-MiniLM-L6-v2
- HuggingFace Embeddings
- Streamlit
- Llama 3 via Groq

### Retrieval Settings
- Top-k: 5
- Chunk Size: 512
- Overlap: 50
""")

# ---------------------------------------------------
# LOAD EMBEDDINGS
# ---------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ---------------------------------------------------
# LOAD LLM — Groq Llama 3 (free)
# ---------------------------------------------------

@st.cache_resource(show_spinner="Connecting to Groq…")
def load_llm():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

embeddings = load_embeddings()
llm        = load_llm()

# ---------------------------------------------------
# LOAD VECTORSTORE
# ---------------------------------------------------

def load_vectorstore():
    index_file = os.path.join(VECTOR_DIR, "index.faiss")
    if not os.path.exists(index_file):
        return None
    try:
        return FAISS.load_local(
            VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load vectorstore: {e}")
        return None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

# ---------------------------------------------------
# SAMPLE PDF LOADER (for recruiters — no upload needed)
# ---------------------------------------------------

st.info("💡 **Demo tip:** Upload any PDF, or click below to load a sample AI research paper instantly.")

if st.button("⚡ Load Sample PDF (Attention Is All You Need)"):
    import urllib.request
    sample_path = os.path.join(UPLOAD_DIR, "attention_paper.pdf")
    with st.spinner("Downloading sample paper…"):
        urllib.request.urlretrieve(
            "https://arxiv.org/pdf/1706.03762",
            sample_path
        )
        loader = PyPDFLoader(sample_path)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source"] = "attention_paper.pdf"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_DIR)
        st.session_state.vectorstore = vectorstore

    st.success(f"✅ Sample loaded! {len(chunks)} chunks indexed.")
    st.markdown("**Try asking:** `What is the transformer architecture?` or `What were the BLEU score results?`")

st.markdown("---")

# ---------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------

st.subheader("📤 Upload Your Own PDFs")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------------------------------------------
# BUILD KNOWLEDGE BASE
# ---------------------------------------------------

if st.button("Build Knowledge Base"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    with st.spinner("Processing PDFs and building vector index…"):

        all_docs = []

        for uploaded_file in uploaded_files:
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(save_path)
            docs   = loader.load()

            for doc in docs:
                doc.metadata["source"] = uploaded_file.name

            all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(all_docs)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_DIR)
        st.session_state.vectorstore = vectorstore

    st.success(f"✅ Knowledge base built with {len(chunks)} chunks from {len(uploaded_files)} file(s).")

# ---------------------------------------------------
# QUERY SECTION
# ---------------------------------------------------

st.subheader("🔍 Ask Questions")

query = st.text_input(
    "Enter your question",
    placeholder="What is this document about?"
)

# ---------------------------------------------------
# SEARCH
# ---------------------------------------------------

if st.button("Search"):

    if st.session_state.vectorstore is None:
        st.error("Please load a sample or build a knowledge base first.")
        st.stop()

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    start_time = time.time()

    # --- FAISS Retrieval (RAG pipeline) ---
    docs = st.session_state.vectorstore.similarity_search_with_score(query, k=5)
    retrieval_end = time.time()

    # --- Build context from retrieved chunks ---
    context       = ""
    citations_map = {}

    for doc, score in docs:
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "?")

        if source not in citations_map:
            citations_map[source] = set()
        citations_map[source].add(page)

        chunk_text = f"\nSOURCE: {source}\nPAGE: {page}\n\nCONTENT:\n{doc.page_content}\n"

        if len(context) + len(chunk_text) > MAX_CONTEXT_CHARS:
            break

        context += chunk_text

    # --- Prompt ---
    prompt = SYSTEM_PROMPT.format(context=context, question=query)

    # --- Groq Llama 3 generation ---
    # AFTER
    # --- Groq generation ---
    try:
        message = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=512,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful document Q&A assistant. Answer concisely using only the provided context. If the answer is not in the context, say so."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        )
        response = message.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        st.stop()

    total_time     = round(time.time() - start_time, 2)
    retrieval_time = round(retrieval_end - start_time, 2)

    # --- Results ---
    st.markdown("---")
    st.subheader("🧠 Answer")
    st.write(response)

    st.subheader("📚 Source Citations")
    for source, pages in citations_map.items():
        sorted_pages = ", ".join(str(p) for p in sorted(pages))
        st.markdown(f"- **{source}** — pages {sorted_pages}")

    st.subheader("⚡ Query Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieved Chunks", len(docs))
    col2.metric("Retrieval Time",   f"{retrieval_time}s")
    col3.metric("Total Latency",    f"{total_time}s")

    st.markdown("---")
    st.subheader("🔍 Retrieved Context Chunks")

    for idx, (doc, score) in enumerate(docs, start=1):
        source    = doc.metadata.get("source", "Unknown")
        page      = doc.metadata.get("page", "?")
        relevance = round(1 / (1 + score), 3)

        with st.expander(
            f"Chunk {idx} | Relevance: {relevance} | {source} | p.{page}"
        ):
            st.write(doc.page_content)

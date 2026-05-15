# 📄 Document Q&A — RAG Pipeline
> LangChain · FAISS · HuggingFace · Groq | **Status: 🚧 In Progress**

---

## What I'm Building

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that enables natural language querying across **1,000+ PDF documents**.

Not just a chatbot — a full system: PDF ingestion → semantic chunking → vector indexing → retrieval → grounded answer generation with **source citations**.

---

## 🎯 Targets

| Metric | Goal |
|--------|------|
| BLEU Score | 0.81+ |
| Documents | 1K+ PDFs |
| Latency | < 1.2s |
| Retrieval | FAISS flat index, Top-5 chunks |
| Embedding | all-MiniLM-L6-v2 |
| LLM | Llama 3.3 70B via Groq |

---

## 🛠️ Stack

`Python` `LangChain` `FAISS` `HuggingFace` `Groq` `Streamlit` `PyPDF` `sentence-transformers`

---

## 📦 Pipeline

```
PDF Upload → Chunker (512 tokens) → Embeddings → FAISS Index → Top-5 Retrieval → LLM → Answer + Citations
```

---

## 🔗 Live Demo

> Basic Streamlit demo running on Streamlit Community Cloud with a **sample PDF preloaded** (Attention Is All You Need paper)
>
> Full 1K+ document pipeline with BLEU evaluation will replace this once training and optimization is complete

**[▶ Try Demo](https://your-app-url.streamlit.app)**

---

## 📍 Progress

### ✅ Done in Demo Version
- [x] Streamlit UI with file uploader and query input
- [x] Multi-PDF upload and batch ingestion
- [x] Semantic chunking — 512 tokens, 50 overlap
- [x] HuggingFace embeddings (`all-MiniLM-L6-v2`)
- [x] FAISS vector store — build, save, and reload
- [x] Top-5 similarity retrieval with relevance scores
- [x] Context injection into LLM prompt (RAG)
- [x] Llama 3.3 70B answer generation via Groq API
- [x] Source citations — document name + page number
- [x] Query metrics — retrieval time, total latency, chunk count
- [x] Chunk explorer — inspect each retrieved chunk
- [x] One-click sample PDF loader (no upload needed)

### 🚧 In Progress
- [ ] BLEU evaluation framework — building labeled Q&A test set
- [ ] Latency optimization — targeting < 1.2s end-to-end
- [ ] Scale testing — 1K+ document corpus

### 📋 Planned
- [ ] Hybrid retrieval — BM25 sparse + FAISS dense fusion
- [ ] Re-ranking layer — cross-encoder on top-k results
- [ ] Streaming responses — token-by-token generation
- [ ] Docker containerization — one-command deployment
- [ ] MLflow experiment tracking

---

## 👤 Author

**Anish Kumar** — ML Engineer (BTech AI/ML)

[GitHub](https://github.com/anishk-neuroforge) · [LinkedIn](https://linkedin.com/in/anishkumar25) · [Portfolio](https://anishk-neuroforge.github.io)

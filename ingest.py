import os
import argparse
import pathlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------------
# LOAD PDF DOCUMENTS
# -----------------------------------

def load_documents(data_dir):

    documents = []

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {data_dir}")

    print(f"\nFound {len(pdf_files)} PDF file(s)\n")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        print(f"  Loading: {pdf_file}")

        try:
            loader = PyPDFLoader(pdf_path)
            docs   = loader.load()

            for doc in docs:
                doc.metadata["source"] = pdf_file

            documents.extend(docs)
            print(f"  → {len(docs)} pages loaded")

        except Exception as e:
            print(f"  ✗ Failed to load {pdf_file}: {e}")

    return documents


# -----------------------------------
# CHUNK DOCUMENTS
# -----------------------------------

def chunk_documents(documents, chunk_size, chunk_overlap):

    print(f"\nChunking {len(documents)} pages…")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"→ {len(chunks)} chunks created")
    return chunks


# -----------------------------------
# CREATE EMBEDDINGS
# -----------------------------------

def load_embedding_model():
    print("\nLoading embedding model…")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------------
# BUILD FAISS INDEX
# -----------------------------------

def build_faiss_index(chunks, embeddings, output_dir):
    print("\nBuilding FAISS index…")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"Saving to: {output_dir}")
    vectorstore.save_local(output_dir)
    print("✅ Done")


# -----------------------------------
# MAIN
# -----------------------------------

def main():

    # FIX: default output_dir is absolute, same location as app.py
    default_vector_dir = str(pathlib.Path(__file__).parent / "vectorstore")

    parser = argparse.ArgumentParser(description="Ingest PDFs into FAISS vectorstore")

    parser.add_argument("--data_dir",   type=str, required=True,
                        help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, default=default_vector_dir,
                        help="Directory to save FAISS index")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--overlap",    type=int, default=50)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== RAG Ingestion Pipeline ===\n")

    documents  = load_documents(args.data_dir)
    print(f"\nTotal pages loaded: {len(documents)}")

    chunks     = chunk_documents(documents, args.chunk_size, args.overlap)
    embeddings = load_embedding_model()

    build_faiss_index(chunks, embeddings, args.output_dir)

    print(f"\n✅ Ingestion complete — {len(chunks)} chunks indexed\n")


# -----------------------------------
# ENTRYPOINT
# -----------------------------------

if __name__ == "__main__":
    main()
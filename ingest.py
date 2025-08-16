import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


DOCS_DIR = Path("documents")
DB_DIR = Path("vectorstore")
DB_DIR.mkdir(exist_ok=True)

def load_docs() -> List:
    docs = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        path = DOCS_DIR / fname
        low = fname.lower()
        try:
            if low.endswith(".pdf"):
                loader = PyPDFLoader(str(path))
                for d in loader.load():
                    # Ensure a readable source and 1-based page will be available later
                    d.metadata["source"] = fname
                    docs.append(d)
            elif low.endswith(".docx"):
                loader = Docx2txtLoader(str(path))
                for d in loader.load():
                    d.metadata["source"] = fname
                    docs.append(d)
            elif low.endswith(".txt"):
                loader = TextLoader(str(path), encoding="utf-8")
                for d in loader.load():
                    d.metadata["source"] = fname
                    docs.append(d)
            elif low.endswith(".doc"):
                # Best effort .doc handling via textract (optional)
                try:
                    import textract  # type: ignore
                    text = textract.process(str(path)).decode("utf-8", errors="ignore")
                    from langchain.schema import Document
                    docs.append(Document(page_content=text, metadata={"source": fname}))
                except Exception as e:
                    print(f"[WARN] Could not extract .doc '{fname}'. Convert to DOCX/PDF. Error: {e}")
            else:
                print(f"[SKIP] Unsupported file: {fname}")
        except Exception as e:
            print(f"[WARN] Failed to load {fname}: {e}")
    return docs

def main():
    print("📄 Loading documents from ./documents ...")
    documents = load_docs()
    if not documents:
        print("No documents found. Add PDFs/DOCX/DOC/TXT to ./documents and rerun.")
        return

    print(f"✅ Loaded {len(documents)} docs/pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150, length_function=len)
    chunks = splitter.split_documents(documents)
    print(f"🔍 Created {len(chunks)} chunks.")

    print("🔢 Building embeddings with Ollama (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 Saving FAISS index to ./vectorstore ...")
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(DB_DIR))
    print("✅ Ingestion completed successfully.")

if __name__ == "__main__":
    main()

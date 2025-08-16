# Offline Doc Q&A (Ollama + Streamlit + FAISS)

This project lets you ask questions about your Interface Description Documents (IDD) fully **offline**.
It uses:
- **Ollama** (local LLM) → `llama3:8b`
- **FAISS** for vector search
- **LangChain** for retrieval
- **Streamlit** UI
- Supports **PDF, DOCX, DOC (best effort), TXT**

## Prereqs (Windows)
1. **Python 3.10+** (add to PATH)
2. **Ollama**: https://ollama.com/download
   ```powershell
   ollama pull llama3:8b
   ollama pull nomic-embed-text
   ```
3. `pip install -r requirements.txt`

## Run
1. Put your files into `documents/` (PDF/DOCX/DOC/TXT).
2. Build index
   ```powershell
   python ingest.py
   ```
3. Start UI
   ```powershell
   streamlit run app.py
   ```
4. Open http://localhost:8501 and ask questions.

## Notes
- `.doc` support uses `textract` if available; if extraction fails, convert `.doc` to `.docx` or PDF.
- Re-run `python ingest.py` whenever you add/remove documents.
- Answers include the **source file name** (and page number if PDF).


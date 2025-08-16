import streamlit as st
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

DB_PATH = "vectorstore"

@st.cache_resource
def load_qa():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOllama(
        model="llama3:8b",
        temperature=0.1,
        num_ctx=4096,
    )

    system_instructions = """
    You are a helpful assistant. 

    First, try to answer the question strictly using the context provided. 
    If the context does not help or is unclear, say: "Not found in the document."
    Do NOT hallucinate when context is given.

    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=system_instructions,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa, llm


def format_sources(source_docs):
    lines = []
    for d in source_docs:
        src = d.metadata.get("source") or "Unknown"
        page = d.metadata.get("page")
        link = f"{src}" + (f"[p.{page+1}]" if isinstance(page, int) else "")
        lines.append(link)
    seen, uniq = set(), []
    for s in lines:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq


st.set_page_config(page_title="Doc Q&A (Offline)", page_icon="📄", layout="wide")
st.title("📄 Interface Document Q&A (Offline)")

qa_chain, llm = load_qa()

with st.sidebar:
    st.header("Search Options")
    k = st.slider("Top matches (k)", min_value=2, max_value=10, value=4, step=1)
    st.caption("Increase if answers seem incomplete.")

query = st.text_input("💬 Your question:", placeholder="e.g., What is the authentication flow for Interface ABC?")

if st.button("Search") and query.strip():
    with st.spinner("Searching documents..."):
        qa_chain.retriever.search_kwargs["k"] = k
        result = qa_chain.invoke({"query": query})

    answer = result["result"].strip()

    # ✅ If not answered properly from context → fallback to general knowledge
    if "Not found in the document" in answer or answer.lower().startswith("i don't know"):
        st.warning("⚠️ Not found in documents. Fetching from general knowledge...")
        fallback_prompt = f"Explain in detail: {query}"
        answer = llm.invoke(fallback_prompt).content

    # Show Answer
    st.subheader("💡 Answer")
    st.write(answer)

    # Show Sources only if relevant
    sources = format_sources(result.get("source_documents", []))
    if sources and "Not found in the document" not in answer:
        st.subheader("📚 Sources")
        for s in sources:
            st.markdown(f"- `{s}`")
    else:
        st.info("No matching sources found in your documents.")

    # Add timestamp
    st.caption(f"🕒 Answered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import os
import re
import shutil
import pandas as pd
import kagglehub
import streamlit as st
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_DB_DIR = "./chroma_db"
LOCAL_CSV_PATH = "minerals.csv"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

CORE_TEXT_COLS = ['Name', 'Crystal Structure', 'Mohs Hardness', 'Specific Gravity', 'Diaphaneity', 'Optical', 'Refractive Index', 'Dispersion']
METADATA_COLS = ['Name', 'Crystal Structure', 'Mohs Hardness', 'Specific Gravity', 'Calculated Density', 'Molar Mass', 'Molar Volume']
IGNORE_COLS = ['Unnamed: 0', 'count']

MINERAL_ALIASES = {
    "lapis lazuli": "Lazurite",
    "ruby": "Corundum",
    "sapphire": "Corundum",
    "emerald": "Beryl",
    "boulder opal": "Opal",
    "amethyst": "Quartz"
}

def format_value_with_units(col, val):
    """Appends explicit physical units to relevant numerical mineral properties."""
    val_str = str(val).strip()
    if col == 'Mohs Hardness':
        return f"{val_str} (Mohs scale)"
    elif col in ['Specific Gravity', 'Calculated Density']:
        return f"{val_str} g/cm³"
    elif col == 'Molar Mass':
        return f"{val_str} g/mol"
    return val_str

def preprocess_query(query: str) -> str:
    """Pre-processes user query by mapping commercial gem/rock aliases to formal mineral names."""
    processed_query = query
    for alias, formal_name in MINERAL_ALIASES.items():
        pattern = re.compile(re.escape(alias), re.IGNORECASE)
        if pattern.search(processed_query):
            processed_query = pattern.sub(f"{alias} ({formal_name})", processed_query)
    return processed_query

@st.cache_resource(show_spinner="Initializing vector store...")
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            _ = vectorstore.similarity_search("test", k=1)
            return vectorstore
        except Exception:
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)

    if os.path.exists(LOCAL_CSV_PATH):
        df = pd.read_csv(LOCAL_CSV_PATH)
    else:
        path = kagglehub.dataset_download("paultimothymooney/minerals-dataset")
        csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in downloaded dataset path.")
        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)

    documents = []
    dynamic_chem_cols = df.iloc[:, 9:135].columns.tolist()

    for _, row in df.iterrows():
        core_parts = []
        for col in CORE_TEXT_COLS:
            val = row.get(col)
            if pd.notna(val) and str(val).strip() != "":
                try:
                    if float(val) == 0:
                        continue
                except (ValueError, TypeError):
                    pass
                formatted_val = format_value_with_units(col, val)
                core_parts.append(f"{col}: {formatted_val}")

        chem_parts = []
        for col in dynamic_chem_cols:
            val = row.get(col)
            if pd.notna(val):
                try:
                    num_val = float(val)
                    if num_val > 0:
                        chem_parts.append(f"{col}: {num_val}")
                except (ValueError, TypeError):
                    continue

        chem_str = ", ".join(chem_parts)
        if chem_str:
            core_parts.append(f"Chemical Composition: {chem_str}")

        page_content = ". ".join(core_parts)

        metadata = {}
        for col in METADATA_COLS:
            val = row.get(col)
            if pd.notna(val) and str(val).strip() != "":
                formatted_val = format_value_with_units(col, val)
                if isinstance(formatted_val, (int, float, str)):
                    metadata[col] = formatted_val
                else:
                    metadata[col] = str(formatted_val)

        documents.append(Document(page_content=page_content, metadata=metadata))

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )

def format_docs(docs):
    formatted = []
    for doc in docs:
        meta_info = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items() if pd.notna(v))
        formatted.append(f"{doc.page_content} | Metadata: {meta_info}")
    return "\n\n".join(formatted)

@st.cache_resource(show_spinner="Initializing RAG components...")
def get_rag_components():
    vectorstore = get_vectorstore()

    # 1. Similarity score threshold retriever (threshold: 0.4, k: 3) using cosine distance space
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.4, "k": 3}
    )

    llm = ChatOllama(model="phi3", temperature=0)

    # 2. Strict RAG Prompt Template (when retrieved context exists)
    strict_rag_template = """You are an expert mineralogy assistant. Answer the question based ONLY on the provided context.

CRITICAL TERMINOLOGY TRANSLATION MAPPINGS:
When generating Chinese responses, you MUST strictly use the following exact mineralogical translations:
- Mohs Hardness -> 莫氏硬度
- Refractive Index -> 折射率
- Crystal Structure -> 晶體結構
- Specific Gravity -> 比重
- Diaphaneity -> 透明度
- Calculated Density -> 計算密度
- Molar Mass -> 莫耳質量
- Chemical Composition -> 化學成分

FORMATTING RULES:
1. When retrieving mineral properties or details, output using a clear Bullet Points (條列式) format.
2. Do NOT include any introductory prose, conversational filler, or concluding sentences before or after the bullet points.

Context:
{context}

Question: {question}
"""
    strict_prompt = ChatPromptTemplate.from_template(strict_rag_template)
    strict_chain = strict_prompt | llm | StrOutputParser()

    # 3. General Knowledge Prompt Template (when retrieved context is empty)
    general_knowledge_template = """⚠️ 以下為通用科學常識，非資料庫精準數據：

Answer the question based on your general knowledge.

Question: {question}
"""
    general_prompt = ChatPromptTemplate.from_template(general_knowledge_template)
    general_chain = general_prompt | llm | StrOutputParser()

    return retriever, strict_chain, general_chain

def get_response_stream(query: str):
    retriever, strict_chain, general_chain = get_rag_components()
    processed_query = preprocess_query(query)
    docs = retriever.invoke(processed_query)

    if docs:
        formatted_context = format_docs(docs)
        return strict_chain.stream({"context": formatted_context, "question": processed_query})
    else:
        return general_chain.stream({"question": processed_query})

def main():
    st.set_page_config(page_title="Mineral Database RAG", page_icon="💎", layout="wide")
    st.title("💎 Mineral Database Local RAG System")
    st.markdown("Explore mineral properties, chemical compositions, and optical attributes using local Ollama (Phi-3), HuggingFace Embeddings, and Chroma DB.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Enter your mineral question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                response = st.write_stream(get_response_stream(user_input))
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)

if __name__ == "__main__":
    main()

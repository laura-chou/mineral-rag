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
REJECTION_MESSAGE = "⚠️ **資料庫中查無此礦物的精確數據。為確保物理與化學參數之嚴謹性，系統拒絕回答。**"

CORE_TEXT_COLS = ['Name', 'Crystal Structure', 'Mohs Hardness', 'Specific Gravity', 'Diaphaneity', 'Optical', 'Refractive Index', 'Dispersion']
METADATA_COLS = ['Name', 'Crystal Structure', 'Mohs Hardness', 'Specific Gravity', 'Calculated Density', 'Molar Mass', 'Molar Volume']
IGNORE_COLS = ['Unnamed: 0', 'count']

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

def extract_target_mineral(translated_query: str, valid_names: list) -> str | None:
    """Matches the LLM-translated English query against valid mineral names."""
    query_lower = translated_query.lower()

    sorted_names = sorted([str(n) for n in valid_names if pd.notna(n)], key=len, reverse=True)

    for name in sorted_names:
        name_str = str(name).strip()
        if not name_str:
            continue

        if re.search(rf'\b{re.escape(name_str)}\b', query_lower, re.IGNORECASE):
            return name_str

        if len(name_str) >= 4 and name_str.lower() in query_lower:
            return name_str

    return None

@st.cache_resource(show_spinner="Initializing vector store...")
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if os.path.exists(LOCAL_CSV_PATH):
        df = pd.read_csv(LOCAL_CSV_PATH)
    else:
        path = kagglehub.dataset_download("paultimothymooney/minerals-dataset")
        csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in downloaded dataset path.")
        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)

    mineral_names = df['Name'].dropna().unique().tolist()

    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            _ = vectorstore.similarity_search("test", k=1)
            return vectorstore, mineral_names
        except Exception:
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)

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
                try:
                    if float(val) == 0:
                        continue
                except (ValueError, TypeError):
                    pass

                formatted_val = format_value_with_units(col, val)
                if isinstance(formatted_val, (int, float, str)):
                    metadata[col] = formatted_val
                else:
                    metadata[col] = str(formatted_val)

        documents.append(Document(page_content=page_content, metadata=metadata))

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    return vectorstore, mineral_names

def format_docs(docs):
    formatted = []
    for doc in docs:
        meta_info = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items() if pd.notna(v))
        formatted.append(f"{doc.page_content} | Metadata: {meta_info}")
    return "\n\n".join(formatted)

@st.cache_resource(show_spinner="Initializing RAG components...")
def get_rag_components():
    vectorstore, mineral_names = get_vectorstore()
    llm = ChatOllama(model="phi3", temperature=0)

    # --- Stage 1: Translation and Name Extraction Chain ---
    translation_template = """You are a mineralogy term extractor. Your ONLY job is to extract the target mineral or gemstone from the user's query and output its formal English mineralogical name.
If the user uses Chinese (e.g. '青金石', '紅寶石') or commercial names (e.g. 'Lapis Lazuli', 'Ruby'), translate them to formal names ('Lazurite', 'Corundum').
Output EXACTLY the English mineral name and nothing else. No punctuation, no explanation.

Query: {query}
Formal English Name:"""
    translator_prompt = ChatPromptTemplate.from_template(translation_template)
    translator_chain = translator_prompt | llm | StrOutputParser()

    # --- Stage 2: Strict Generation Chain ---
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
1. When the requested properties exist in the Context, output them using a clear Bullet Points (條列式) format.
2. Do NOT include any introductory prose, conversational filler, or concluding sentences when data IS successfully found.
3. ELEGANT MISSING DATA HANDLING (缺失資料優化):
   - Do NOT output repetitive bullet lines saying "資料庫無此數據" for each missing attribute.
   - If the requested properties/attributes are missing from the Context, combine them into ONE natural, polite, and unified sentence.
   - Include the mineral's formal English name and its common Traditional Chinese name in parentheses if applicable (e.g., 'Boulder opal（礫石蛋白石）', 'Lapis Lazuli（青金石）').
   - Standard Response Template:
     "抱歉，資料庫中目前沒有收錄 [礦物英文名]（[中文俗名]）的任何相關紀錄，因此無法為您提供其 [缺失屬性A] 與 [缺失屬性B] 的數據。"

Context:
{context}

Question: {question}
"""
    strict_prompt = ChatPromptTemplate.from_template(strict_rag_template)
    strict_chain = strict_prompt | llm | StrOutputParser()

    return vectorstore, translator_chain, strict_chain, mineral_names

def get_response_stream(query: str):
    vectorstore, translator_chain, strict_chain, mineral_names = get_rag_components()

    # 1. First stage: Translate/extract formal English mineral name using LLM
    translated_query = translator_chain.invoke({"query": query}).strip()

    # 2. Second stage: Python Gatekeeper matching translated name against CSV records
    target_mineral = extract_target_mineral(translated_query, mineral_names)
    if not target_mineral:
        def empty_response():
            yield REJECTION_MESSAGE
        return empty_response()

    # 3. Third stage: Filter Chroma vector store strictly by Name metadata
    docs = vectorstore.similarity_search(query, k=3, filter={"Name": target_mineral})
    if not docs:
        def empty_response():
            yield REJECTION_MESSAGE
        return empty_response()

    # 4. Fourth stage: Stream RAG answer from strict_chain
    formatted_context = format_docs(docs)
    return strict_chain.stream({"context": formatted_context, "question": query})

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

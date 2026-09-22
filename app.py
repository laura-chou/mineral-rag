import os
import re
import shutil
import pandas as pd
import kagglehub
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

def get_or_create_vectorstore(embeddings):
    """
    Checks if Chroma DB exists locally and tests embedding dimension compatibility.
    If incompatible or missing, rebuilds the vector store using local minerals.csv or kagglehub.
    Uses collection_metadata={"hnsw:space": "cosine"} to enforce cosine similarity relevance scoring.
    """
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print("✓ Checking existing vector store in ./chroma_db ...")
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            _ = vectorstore.similarity_search("test", k=1)
            print("✓ Vector store loaded successfully with cosine distance metric.")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Vector store dimension mismatch or corruption detected ({e}). Removing old database...")
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)

    print("✓ Initializing ingestion pipeline for multilingual vector store...")

    if os.path.exists(LOCAL_CSV_PATH):
        print("✓ Local minerals.csv detected. Loading directly...")
        df = pd.read_csv(LOCAL_CSV_PATH)
    else:
        print("Local minerals.csv not found. Attempting to download via kagglehub...")
        path = kagglehub.dataset_download("paultimothymooney/minerals-dataset")
        csv_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in downloaded dataset path.")
        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)

    print(f"Loaded full dataset with total {len(df)} rows.")

    print("Constructing documents with categorized text, units, metadata, and chemical composition (cols J:EE)...")
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

    print(f"Created {len(documents)} structured Document objects.")

    print("Generating multilingual HuggingFace embeddings and persisting into Chroma vector store with Cosine space...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("✓ Multilingual Chroma DB successfully created and persisted with cosine similarity metric.")
    return vectorstore

def format_docs(docs):
    formatted = []
    for doc in docs:
        meta_info = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items() if pd.notna(v))
        formatted.append(f"{doc.page_content} | Metadata: {meta_info}")
    return "\n\n".join(formatted)

def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = get_or_create_vectorstore(embeddings)

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

    print("\n" + "="*50)
    print(" Mineral Database RAG Retrieval System (CLI)")
    print("="*50)
    print("System ready! You can ask your questions at any time.")

    while True:
        try:
            user_input = input("\nEnter your mineral question (type 'exit' or 'quit' to leave): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nThank you for using the Mineral RAG system. Goodbye!")
                break

            processed_query = preprocess_query(user_input)
            print(f"\nSearching for: {processed_query}...")

            # Retrieve documents using similarity score threshold
            retrieved_docs = retriever.invoke(processed_query)

            print("\nResponse:")
            if retrieved_docs:
                print("(Database context retrieved)")
                formatted_context = format_docs(retrieved_docs)
                response = strict_chain.invoke({"context": formatted_context, "question": processed_query})
                print(response)
            else:
                print("(No database context above threshold. Falling back to general knowledge)")
                response = general_chain.invoke({"question": processed_query})
                print(response)

        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()

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

def get_or_create_vectorstore(embeddings):
    """
    Loads or creates Chroma DB vector store and returns (vectorstore, mineral_names).
    Filters out zero or 0.0 values from metadata attributes.
    """
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
        print("✓ Checking existing vector store in ./chroma_db ...")
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings,
                collection_metadata={"hnsw:space": "cosine"}
            )
            _ = vectorstore.similarity_search("test", k=1)
            print("✓ Vector store loaded successfully.")
            return vectorstore, mineral_names
        except Exception as e:
            print(f"⚠️ Vector store dimension mismatch or corruption detected ({e}). Removing old database...")
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)

    print("✓ Initializing ingestion pipeline for multilingual vector store...")
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

    print(f"Created {len(documents)} structured Document objects.")

    print("Generating multilingual HuggingFace embeddings and persisting into Chroma vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("✓ Multilingual Chroma DB successfully created and persisted.")
    return vectorstore, mineral_names

def format_docs(docs):
    formatted = []
    for doc in docs:
        meta_info = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items() if pd.notna(v))
        formatted.append(f"{doc.page_content} | Metadata: {meta_info}")
    return "\n\n".join(formatted)

def get_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore, mineral_names = get_or_create_vectorstore(embeddings)
    llm = ChatOllama(model="phi3", temperature=0)

    # --- Stage 1: Translation and Name Extraction Chain ---
    translation_template = """You are a mineralogy term extractor. Your ONLY job is to extract the target mineral or gemstone from the user's query and output its formal English mineralogical name.
If the user uses Chinese (e.g. '青金石', '紅寶石') or commercial names (e.g. 'Lapis Lazuli', 'Ruby'), translate them to formal names ('Lazurite', 'Corundum').
Output EXACTLY the English mineral name and nothing else. No punctuation, no explanation.

Query: {query}
Formal English Name:"""
    translator_prompt = ChatPromptTemplate.from_template(translation_template)
    translator_chain = translator_prompt | llm | StrOutputParser()

    # --- Stage 2: Strict Generation Chain (Pure English Prompt) ---
    strict_rag_template = """You are an expert mineralogy assistant. Answer the question based ONLY on the provided context.

STRICT GENERATION RULES:
1. Do NOT include any greetings (e.g., 'Hello', 'Hi'), introductory prose, conversational filler, or concluding remarks. Start directly with the data.
2. If the requested property exists in the Context, output it using a concise Bullet Points format.
3. If a requested property is missing or invalid in the Context, output:
   - [Property Name]: Data unavailable in database
4. NEVER invent, hallucinate, or assume any properties not explicitly stated in the Context.

Context:
{context}

Question: {question}
"""
    strict_prompt = ChatPromptTemplate.from_template(strict_rag_template)
    strict_chain = strict_prompt | llm | StrOutputParser()

    return vectorstore, translator_chain, strict_chain, mineral_names

def main():
    vectorstore, translator_chain, strict_chain, mineral_names = get_rag_components()

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

            print("\n[Stage 1] Extracting formal mineral name...")
            translated_query = translator_chain.invoke({"query": user_input}).strip()
            print(f"Extracted English Term: {translated_query}")

            print("[Stage 2] Gatekeeper check against CSV records...")
            target_mineral = extract_target_mineral(translated_query, mineral_names)
            if not target_mineral:
                print("\nResponse:")
                print(REJECTION_MESSAGE)
                continue

            print(f"[Stage 3] Filtering Chroma DB for '{target_mineral}'...")
            docs = vectorstore.similarity_search(user_input, k=3, filter={"Name": target_mineral})
            if not docs:
                print("\nResponse:")
                print(REJECTION_MESSAGE)
                continue

            print("[Stage 4] Generating grounded answer...")
            print("\nResponse:")
            formatted_context = format_docs(docs)
            response = strict_chain.invoke({"context": formatted_context, "question": user_input})
            print(response)

        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()

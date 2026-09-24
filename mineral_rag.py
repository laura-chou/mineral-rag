import os
import re
import shutil
from datetime import datetime
import pandas as pd
import kagglehub
from thefuzz import fuzz, process
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_DB_DIR = "./chroma_db"
LOCAL_CSV_PATH = "minerals.csv"
LOG_FILE_PATH = "missing_minerals.txt"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
REJECTION_MESSAGE = "⚠️ **Exact data for this mineral is not found in the database. To ensure physical and chemical accuracy, the system declines to answer.**"

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

def log_missing_mineral(raw_query: str, extracted_term: str = ""):
    """Logs raw query, extracted term, and timestamp to missing_minerals.txt."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f'[{timestamp}] Raw Query: "{raw_query}" | Extracted Term: "{extracted_term}"\n'
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        print(f"Failed to log missing mineral: {e}")

def extract_target_mineral(translated_query: str, valid_names: list) -> str | None:
    """Matches the LLM-translated English query against valid mineral names using exact regex and fuzz.WRatio fuzzy matching."""
    query_lower = translated_query.lower().strip()
    if not query_lower:
        return None

    clean_valid_names = [str(n).strip() for n in valid_names if pd.notna(n) and str(n).strip() != ""]
    sorted_names = sorted(clean_valid_names, key=len, reverse=True)

    # 1. Exact Word Boundary Regex Match
    for name_str in sorted_names:
        if re.search(rf'\b{re.escape(name_str)}\b', query_lower, re.IGNORECASE):
            return name_str

        if len(name_str) >= 4 and name_str.lower() in query_lower:
            return name_str

    # 2. Optimized Fuzzy String Match using process.extractOne with fuzz.WRatio (threshold score >= 80)
    best_match = process.extractOne(query_lower, clean_valid_names, scorer=fuzz.WRatio)
    if best_match and best_match[1] >= 80:
        return best_match[0]

    return None

def get_or_create_vectorstore(embeddings):
    """
    Loads or creates Chroma DB vector store and returns (vectorstore, mineral_names).
    Uses df.to_dict('records') for fast row iteration preserving exact column names with spaces.
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

    print("Constructing documents with fast to_dict('records') iteration (cols J:EE)...")
    documents = []

    dynamic_chem_cols = df.iloc[:, 9:135].columns.tolist()

    # Fast iteration using to_dict('records') preserving exact column names with spaces
    for row_dict in df.to_dict('records'):
        core_parts = []
        for col in CORE_TEXT_COLS:
            val = row_dict.get(col)
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
            val = row_dict.get(col)
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
            val = row_dict.get(col)
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

    # --- Stage 1: Minimal Translation and Name Extraction Chain ---
    translation_template = """You are an expert mineralogy translator.
Identify and extract the primary mineral or gemstone name from the query. If it is in a foreign language, translate it to its English mineral name.
Output EXACTLY the formal English name and nothing else.

Query: {query}
Formal English Name:"""
    translator_prompt = ChatPromptTemplate.from_template(translation_template)
    translator_chain = translator_prompt | llm | StrOutputParser()

    # --- Stage 2: Strict Generation Chain with Laser Focus Rule ---
    strict_rag_template = """You are an expert mineralogy assistant. Answer the question based ONLY on the provided context.

STRICT GENERATION RULES:
1. LASER FOCUS: Output ONLY the exact property or information specifically requested in the Question. Do NOT output unrequested properties or dump the entire context. Do NOT include greetings, introductory prose, conversational filler, or concluding remarks. Start directly with the requested data.
2. If the requested property exists in the Context, output it using a concise Bullet Points format.
3. If a requested property is missing or invalid in the Context, output:
   - [Property Name]: not specified
4. NEVER invent, hallucinate, or assume any properties not explicitly stated in the Context.

Context:
{context}

Question: {question}
"""
    strict_prompt = ChatPromptTemplate.from_template(strict_rag_template)
    strict_chain = strict_prompt | llm | StrOutputParser()

    return vectorstore, translator_chain, strict_chain, mineral_names

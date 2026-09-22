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
    Uses column J (idx 9) to EE (idx 135) for chemical composition attributes.
    """
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print("✓ Checking existing vector store in ./chroma_db ...")
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=embeddings
            )
            # Dimension test query to ensure compatibility with paraphrase-multilingual-MiniLM-L12-v2
            _ = vectorstore.similarity_search("test", k=1)
            print("✓ Vector store loaded successfully and dimension verified.")
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

    # Extract chemical composition columns strictly from column index 9 (J) to 135 (EE)
    dynamic_chem_cols = df.iloc[:, 9:135].columns.tolist()

    for _, row in df.iterrows():
        # 1. Build Core Text Part (skipping NaN, empty strings, and zero values)
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

        # 2. Build Dynamic Chemical Composition Part from J to EE columns (> 0)
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

        # 3. Build Metadata Dict with units
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

    # Local Embedding & Storage
    print("Generating multilingual HuggingFace embeddings and persisting into Chroma vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print("✓ Multilingual Chroma DB successfully created and persisted.")
    return vectorstore

def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = get_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # RAG Retrieval Loop & Inference Chain with Strict Guardrails and Bullet Points Formatting
    print("Initializing ChatOllama Phi-3 LLM chain...")
    llm = ChatOllama(model="phi3", temperature=0)

    template = """You are an expert mineralogy assistant. Answer the question based ONLY on the provided context.

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
3. If the context does not contain enough information to answer the question, state EXACTLY:
"我無法根據提供的上下文回答這個問題。"

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        formatted = []
        for doc in docs:
            meta_info = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items() if pd.notna(v))
            formatted.append(f"{doc.page_content} | Metadata: {meta_info}")
        return "\n\n".join(formatted)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

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
            response = rag_chain.invoke(processed_query)
            print("\nResponse:")
            print(response)
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()

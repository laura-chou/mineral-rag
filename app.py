import os
import re
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
    Checks if Chroma DB exists locally.
    If it exists and contains files, loads it directly.
    Otherwise, loads local minerals.csv or downloads via kagglehub,
    processes full dataset documents with categorized text and units, and persists to Chroma DB.
    """
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print("✓ Loading existing vector store from ./chroma_db ...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        return vectorstore

    print("✓ Vector store not found locally. Initializing ingestion pipeline...")

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

    print("Constructing documents with categorized text, units, metadata, and dynamic composition...")
    documents = []

    # Identify dynamic chemical composition columns
    dynamic_chem_cols = [
        col for col in df.columns
        if col not in CORE_TEXT_COLS and col not in METADATA_COLS and col not in IGNORE_COLS
    ]

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

        # 2. Build Dynamic Chemical Composition Part (> 0)
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
    print("Generating HuggingFace embeddings and persisting into Chroma vector store...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print("✓ Chroma DB successfully created and persisted.")
    print("✓ Vector store updated with non-zero filtered properties.")
    return vectorstore

def main():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = get_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # RAG Retrieval Loop & Inference Chain with Strict Guardrails and Markdown Table Formatting
    print("Initializing ChatOllama Phi-3 LLM chain...")
    llm = ChatOllama(model="phi3", temperature=0)

    template = """You are an expert assistant for a mineral database.
Answer the question based ONLY on the following provided context.
If multiple physical or optical properties are requested or available, present them cleanly in a Markdown table.
If the context does not contain enough information to answer the question, explicitly state:
"I cannot answer this question based on the provided context."
Do not invent or extrapolate any information beyond what is strictly stated in the context.

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

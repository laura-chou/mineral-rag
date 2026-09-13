import os
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

CORE_TEXT_COLS = ['Name', 'Crystal Structure', 'Diaphaneity', 'Optical', 'Refractive Index', 'Dispersion']
METADATA_COLS = ['Name', 'Crystal Structure', 'Mohs Hardness', 'Specific Gravity', 'Calculated Density', 'Molar Mass', 'Molar Volume']
IGNORE_COLS = ['Unnamed: 0', 'count']

def get_or_create_vectorstore(embeddings):
    """
    Checks if Chroma DB exists locally.
    If it exists and contains files, loads it directly.
    Otherwise, checks for local CSV file 'minerals.csv' or falls back to kagglehub download,
    then processes documents with categorized text and metadata, and persists to Chroma DB.
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

    print(f"Loaded dataset with total {len(df)} rows.")

    # Memory Optimization: Slice top 100 rows
    df = df.head(100)
    print(f"Sliced top {len(df)} rows for memory optimization.")

    print("Constructing documents with categorized text, metadata, and dynamic composition...")
    documents = []

    # Identify dynamic chemical composition columns
    dynamic_chem_cols = [
        col for col in df.columns
        if col not in CORE_TEXT_COLS and col not in METADATA_COLS and col not in IGNORE_COLS
    ]

    for _, row in df.iterrows():
        # 1. Build Core Text Part
        core_parts = []
        for col in CORE_TEXT_COLS:
            val = row.get(col)
            if pd.notna(val) and str(val).strip() != "":
                core_parts.append(f"{col}: {val}")

        # 2. Build Dynamic Chemical Composition Part (> 0)
        chem_parts = []
        for col in dynamic_chem_cols:
            val = row.get(col)
            if pd.notna(val):
                try:
                    num_val = float(val)
                    if num_val > 0:
                        chem_parts.append(f"{col}: {num_val}")
                except ValueError:
                    continue

        chem_str = ", ".join(chem_parts)
        if chem_str:
            core_parts.append(f"Chemical Composition: {chem_str}")

        page_content = ". ".join(core_parts)

        # 3. Build Metadata Dict
        metadata = {}
        for col in METADATA_COLS:
            val = row.get(col)
            if pd.notna(val) and str(val).strip() != "":
                if isinstance(val, (int, float, str)):
                    metadata[col] = val
                else:
                    metadata[col] = str(val)

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
    return vectorstore

def main():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = get_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # RAG Retrieval Loop & Inference Chain with Strict Guardrails
    print("Initializing ChatOllama Phi-3 LLM chain...")
    llm = ChatOllama(model="phi3", temperature=0)

    template = """You are an expert assistant for a mineral database.
Answer the question based ONLY on the following provided context.
If the context does not contain enough information to answer the question, explicitly state:
"I cannot answer this question based on the provided context."
Do not invent or extrapolate any information beyond what is strictly stated in the context.

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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

            print("\nSearching...")
            response = rag_chain.invoke(user_input)
            print("\nResponse:")
            print(response)
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")

if __name__ == "__main__":
    main()

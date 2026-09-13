import os
import pandas as pd
import kagglehub
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_DB_DIR = "./chroma_db"
LOCAL_CSV_PATH = "minerals.csv"

def get_or_create_vectorstore(embeddings):
    """
    Checks if Chroma DB exists locally.
    If it exists and contains files, loads it directly.
    Otherwise, checks for local CSV file 'minerals.csv' or falls back to kagglehub download,
    then processes documents and persists to Chroma DB.
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

    # Data Transformation
    print("Constructing 'mineral_description' column...")
    df['mineral_description'] = df.apply(
        lambda row: f"Mineral Name: {row.get('name', 'N/A')}. "
                    f"Formula: {row.get('formula', 'N/A')}. "
                    f"Properties: {row.get('properties', 'N/A')}",
        axis=1
    )

    loader = DataFrameLoader(df, page_content_column="mineral_description")
    documents = loader.load()

    # Local Embedding & Storage (Direct passage without chunk splitting)
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

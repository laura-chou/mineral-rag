import os
import pandas as pd
import kagglehub
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    print("Step 1: Downloading dataset via kagglehub...")
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

    # Step 2: Data Transformation
    print("Step 2: Constructing 'mineral_description' column...")
    df['mineral_description'] = df.apply(
        lambda row: f"Mineral Name: {row.get('name', 'N/A')}. "
                    f"Formula: {row.get('formula', 'N/A')}. "
                    f"Properties: {row.get('properties', 'N/A')}",
        axis=1
    )

    loader = DataFrameLoader(df, page_content_column="mineral_description")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"Split dataset into {len(docs)} text chunks.")

    # Step 3: Local Embedding & Storage
    print("Step 3: Generating HuggingFace embeddings and initializing Chroma vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Step 4: RAG Retrieval Loop & Inference
    print("Step 4: Initializing ChatOllama Phi-3 LLM chain...")
    llm = ChatOllama(model="phi3", temperature=0)

    template = """Answer the question based only on the following context:
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

    query = "What are the physical properties and formula of Quartz?"
    print(f"\nUser Query: {query}\n")
    response = rag_chain.invoke(query)
    print("Response:\n", response)

if __name__ == "__main__":
    main()

# Local RAG System for Mineral Database

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20(Phi--3)-black?style=flat&logo=ollama&logoColor=white)](https://ollama.ai/)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-046A38?style=flat)](https://www.trychroma.com/)

An enterprise-grade, privacy-focused Local Retrieval-Augmented Generation (RAG) interactive command-line system engineered for querying mineralogical data. Powered by LangChain, Ollama (`phi3`), HuggingFace Embeddings (`all-MiniLM-L6-v2`), and ChromaDB, this system runs fully offline on edge devices without relying on external cloud LLM APIs.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Architectural Workflow](#key-architectural-workflow)
- [Prerequisites & Setup](#prerequisites--setup)
- [Installation & Quickstart Guide](#installation--quickstart-guide)
- [Project Directory Structure](#project-directory-structure)
- [Configuration & Parameters](#configuration--parameters)
- [Implementation Code Snippet](#implementation-code-snippet)
- [Troubleshooting & Performance Notes](#troubleshooting--performance-notes)

---

## Project Overview

The Local RAG System for Mineral Database provides deterministic, hallucination-resistant query-answering over mineral datasets. Raw data is automatedly ingested via `kagglehub`, transformed into structured text summaries, embedded locally into dense vector spaces, and stored within a persistent Chroma vector database (`./chroma_db`). Upon user query execution, relevant mineral context is retrieved via vector similarity search and fed to a local Phi-3 small language model served by Ollama to synthesize accurate, grounded answers in an interactive CLI loop.

Key Features:
- **100% Air-Gapped Execution:** Operates entirely locally with zero telemetry or data egress to third-party endpoints.
- **Persistent Caching:** Vector embeddings are computed once and cached on disk in `./chroma_db` for near-instantaneous startup on subsequent runs.
- **Record-Level Preservation:** Documents map 1:1 to mineral records to prevent formula/property truncation across chunk boundaries.
- **Interactive CLI Interface:** Supports continuous, interactive user prompts with exit handling (`exit` / `quit`).
- **Strict Guardrails:** Configured to strictly answer from retrieved context and fallback to `"根據現有資料庫，無法回答此問題。"` when context is insufficient.

---

## Key Architectural Workflow

The system processes data linearly through an ingestion pipeline, caches embeddings in a persistent vector database, and executes a deterministic interactive retrieval loop for grounded generation.

```mermaid
graph TD
    %% System Styles
    classDef Ingestion fill:#FFE2E2,stroke:#FF6B6B,stroke-width:2px;
    classDef Transformation fill:#E2F0D9,stroke:#70AD47,stroke-width:2px;
    classDef Storage fill:#DEEBF7,stroke:#4F81BD,stroke-width:2px;
    classDef Inference fill:#FFF2CC,stroke:#FFC000,stroke-width:2px;

    %% Stage 1: Data Ingestion
    subgraph S1 [1. Data Ingestion]
        A[Start Script] --> B{Check Chroma DB Exists?}
        B -- No --> C[Download Dataset via kagglehub]
        C --> D[Load CSV into Pandas DataFrame]
        D --> E[Slice Top 100 Rows <br><i>Memory Optimization</i>]
    end
    class B,C,D,E Ingestion;

    %% Stage 2: Data Transformation
    subgraph S2 [2. Data Transformation]
        E --> F[Construct 'mineral_description' Column]
        F --> G[Convert to Documents via DataFrameLoader]
    end
    class F,G Transformation;

    %% Stage 3: Local Embedding & Storage
    subgraph S3 [3. Local Embedding & Storage]
        G --> H[Generate Vectors via HuggingFace Embeddings <br><i>all-MiniLM-L6-v2</i>]
        H --> I[(Store & Persist in Chroma Vector DB)]
        B -- Yes --> I
        I --> J[Expose as Retriever <br><i>Search Kwargs: k=3</i>]
    end
    class H,I,J Storage;

    %% Stage 4: RAG Retrieval & Inference
    subgraph S4 [4. RAG Retrieval Loop]
        K[Interactive CLI User Prompt] --> L[Vector Similarity Search]
        J -.->|Retrieve Context| L
        L --> M[Inject Context & Strict Prompt Guardrails]
        M --> N[Local Inference via Ollama <br><i>Phi-3 LLM</i>]
        N --> O[Generate Grounded Answer / Fallback]
    end
    class K,L,M,N,O Inference;
```

---

## Prerequisites & Setup

### System Requirements
- **OS:** Linux / macOS / Windows 10 or 11 (WSL2 recommended)
- **Memory:** Minimum 8GB system RAM
- **GPU:** Minimum 4GB VRAM (Optional; CPU-only execution supported)
- **Python:** 3.10 or higher

### Step 1: Install Ollama & Pull Phi-3 Model
Install Ollama from [ollama.com](https://ollama.com) and pull the default quantized Phi-3 model:

```bash
ollama pull phi3
```

Ensure the Ollama service is running in the background before starting the application:

```bash
ollama serve
```

### Step 2: Configure Environment & Virtual Environment
Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

> [!NOTE]
> `kagglehub` automatically manages dataset downloading and caching in `~/.cache/kagglehub`. Public datasets downloaded through `kagglehub` do not require explicit Kaggle API keys or manual authentication.

---

## Installation & Quickstart Guide

### 1. Install Dependencies

Install the required packages using `pip`:

```bash
pip install pandas langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama sentence-transformers kagglehub
```

### 2. Run the Application

Execute the interactive command-line interface:

```bash
python app.py
```

```text
==================================================
 礦物資料庫 RAG 檢索系統 (Mineral RAG CLI)
==================================================
系統就緒！可隨時輸入問題。

請輸入您的礦物問題 (輸入 'exit' 或 'quit' 離開): Quartz 的化學式與物理特性是什麼？
```

> [!IMPORTANT]
> Ensure Ollama is running (`ollama serve`) before executing `python app.py`.

---

## Project Directory Structure

```text
mineral-rag/
│
├── chroma_db/               # Local persistent storage directory for ChromaDB embeddings
├── app.py                   # Main Python application entrypoint execution script
├── README.md                # System technical documentation and workflow specifications
└── requirements.txt         # Declared python dependencies version sheet
```

---

## Configuration & Parameters

The system behavior can be tuned by modifying parameters globally inside the runtime execution file.

| Parameter | Default Value | Target Component | Purpose |
| :--- | :--- | :--- | :--- |
| `CHROMA_DB_DIR` | `./chroma_db` | Vector Store | Target directory for persisting vector embeddings on disk. |
| `df.head()` | `100` | Data Preprocessing | Limits initial dataframe rows to fit low-RAM system bounds. |
| `model_name` | `all-MiniLM-L6-v2` | HuggingFaceEmbeddings | Local sentence transformer model for dense vector generation. |
| `model` | `phi3` | ChatOllama | Target local LLM backend optimized for 4GB VRAM. |
| `search_kwargs`| `{"k": 3}` | Chroma VectorDB | Number of highly relevant context snippets retrieved per query. |
| `temperature`  | `0` | ChatOllama LLM | Set to zero to eliminate creative hallucinations and enforce deterministic output. |

---

## Implementation Code Snippet

Below is the complete implementation showing vector caching, prompt guardrails, and interactive CLI in `app.py`:

```python
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

def get_or_create_vectorstore(embeddings):
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print("✓ Loading existing vector store from ./chroma_db ...")
        return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

    print("Step 1: Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("paultimothymooney/minerals-dataset")
    csv_file = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')][0]
    df = pd.read_csv(csv_file).head(100)

    df['mineral_description'] = df.apply(
        lambda row: f"Mineral Name: {row.get('name', 'N/A')}. "
                    f"Formula: {row.get('formula', 'N/A')}. "
                    f"Properties: {row.get('properties', 'N/A')}",
        axis=1
    )

    loader = DataFrameLoader(df, page_content_column="mineral_description")
    documents = loader.load()

    return Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=CHROMA_DB_DIR)

def main():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = get_or_create_vectorstore(embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOllama(model="phi3", temperature=0)
    template = """Answer the question based ONLY on the following context.
If the context does not contain enough information, state: "根據現有資料庫，無法回答此問題。"

Context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        query = input("\n請輸入您的礦物問題 (輸入 'exit' 或 'quit' 離開): ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if query:
            print(rag_chain.invoke(query))

if __name__ == "__main__":
    main()
```

---

## Troubleshooting & Performance Notes

> [!IMPORTANT]
> **Persistent Caching Advantage:**
> The system automatically detects existing vector files in `./chroma_db`. Subsequent runs bypass dataset downloading and embedding recalculation completely, reducing initial startup time from minutes to milliseconds.

### Common Issues & Mitigation

1. **Ollama Connection Refused (`http://localhost:11434`)**
   - **Cause:** The Ollama background service is not active.
   - **Solution:** Execute `ollama serve` in a separate terminal before running the application script.

2. **Phi-3 Model Not Found**
   - **Cause:** The model binary has not been pulled locally.
   - **Solution:** Run `ollama pull phi3` to download the quantized weights (~2.3GB).

3. **Re-generating Vectors on Every Execution**
   - **Cause:** Deleting or corrupting `./chroma_db`.
   - **Solution:** Keep `./chroma_db` intact so `app.py` loads cached embeddings directly.

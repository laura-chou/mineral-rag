# Local RAG System for Mineral Database

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama%20(Phi--3)-black?style=flat&logo=ollama&logoColor=white)](https://ollama.ai/)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-046A38?style=flat)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

An enterprise-grade, privacy-focused Local Retrieval-Augmented Generation (RAG) system engineered for querying mineralogical data via both an interactive CLI and a modern Streamlit Web UI. Powered by LangChain, Ollama (`phi3`), HuggingFace Multilingual Embeddings (`paraphrase-multilingual-MiniLM-L12-v2`), and ChromaDB, this system runs fully offline on edge devices without relying on external cloud LLM APIs.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Architectural Workflow](#key-architectural-workflow)
- [Prerequisites & Setup](#prerequisites--setup)
- [Installation & Quickstart Guide](#installation--quickstart-guide)
- [Streamlit Web Interface](#streamlit-web-interface)
- [Project Directory Structure](#project-directory-structure)
- [Configuration & Parameters](#configuration--parameters)
- [Implementation Code Snippet](#implementation-code-snippet)
- [Troubleshooting & Performance Notes](#troubleshooting--performance-notes)

---

## Project Overview

The Local RAG System for Mineral Database provides deterministic, hallucination-resistant query-answering over mineral datasets. Raw data is automatedly ingested via local `minerals.csv` (or downloaded via `kagglehub`), transformed into structured `Document` objects with explicit physical units, metadata dictionaries, and chemical composition elements extracted from CSV columns J to EE (indices 9–135), embedded locally into dense vector spaces via `paraphrase-multilingual-MiniLM-L12-v2`, and stored within a persistent Chroma vector database (`./chroma_db`). Upon user query execution, relevant mineral context is retrieved via vector similarity search ($k=3$) and streamed in real-time to a local Phi-3 small language model served by Ollama to synthesize accurate, grounded answers in bullet points.

Key Features:
- **100% Air-Gapped Execution:** Operates entirely locally with zero telemetry or data egress to third-party endpoints.
- **Multilingual Dense Embeddings:** Leverages `paraphrase-multilingual-MiniLM-L12-v2` for cross-lingual semantic vector retrieval.
- **Automatic Dimension Compatibility:** Automatically detects dimension mismatches in existing `./chroma_db` stores and recreates them cleanly.
- **Targeted Chemical Composition Ingestion:** Slices CSV columns J to EE (indices 9 to 135) to capture element composition (> 0).
- **Dual Interface:** Interactive Command-Line Interface (`app.py`) and a real-time streaming Streamlit Web UI (`app_ui.py`).
- **Real-Time Output Streaming:** `st.write_stream` ensures low-latency responsive chat rendering without UI freezing.
- **Bullet-Point Output Format:** System prompt instructs Phi-3 to respond strictly in clean bullet points without conversational filler.
- **Strict Guardrails:** Configured to strictly answer from retrieved context and fallback to `"我無法根據提供的上下文回答這個問題。"` when context is insufficient.

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
        A[Start App CLI / Web UI] --> B{Check Chroma DB & Dimension?}
        B -- Compatible --> L
        B -- Missing/Mismatch --> C{Check Local minerals.csv?}
        C -- Yes --> D[Load minerals.csv via Pandas]
        C -- No --> E[Download Dataset via kagglehub]
        E --> D
        D --> F[Process Full Dataset - 3,112 Rows]
    end
    class B,C,D,E,F Ingestion;

    %% Stage 2: Data Transformation
    subgraph S2 [2. Data Transformation]
        F --> G[Extract CORE_TEXT_COLS + Units]
        F --> H[Extract Chemical Cols J:EE - idx 9:135 > 0]
        F --> I[Extract METADATA_COLS + Units]
        G --> J[Construct Document Objects]
        H --> J
        I --> J
    end
    class G,H,I,J Transformation;

    %% Stage 3: Local Embedding & Storage
    subgraph S3 [3. Local Embedding & Storage]
        J --> K[Generate Multilingual Vectors <br><i>paraphrase-multilingual-MiniLM-L12-v2</i>]
        K --> L[(Store & Persist in Chroma Vector DB)]
        L --> M[Expose as Retriever <br><i>Search Kwargs: k=3</i>]
    end
    class K,L,M Storage;

    %% Stage 4: RAG Retrieval & Inference
    subgraph S4 [4. RAG Retrieval Loop]
        N[CLI Input / Streamlit Chat Input] --> O[Gemology Alias Pre-processing]
        O --> P[Vector Similarity Search - Top K=3]
        M -.->|Retrieve Context & Metadata| P
        P --> Q[Inject Context, Metadata & Bullet-Point Guardrails]
        Q --> R[Local Inference via Ollama <br><i>Phi-3 LLM</i>]
        R --> S[Stream Real-Time Bullet Point Response]
    end
    class N,O,P,Q,R,S Inference;
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
> If a local `minerals.csv` file is present in the repository root directory, `app.py` and `app_ui.py` will load it directly and bypass Kaggle downloading.

---

## Installation & Quickstart Guide

### 1. Install Dependencies

Install the required packages using `pip`:

```bash
pip install pandas langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama sentence-transformers kagglehub streamlit
```

### 2. Run the Command-Line Application

Execute the interactive command-line interface:

```bash
python app.py
```

---

## Streamlit Web Interface

Launch the modern chat UI in your browser:

```bash
streamlit run app_ui.py
```

Features of the Web UI:
- **Real-time Output Streaming:** `st.write_stream` prevents interface freezing and provides low-latency chat updates.
- **Cached Vector Operations:** `@st.cache_resource` prevents re-indexing data on user actions.
- **Automatic Database Migration:** Handles vector dimension changes seamlessly upon model switching.

---

## Project Directory Structure

```text
mineral-rag/
│
├── chroma_db/               # Local persistent storage directory for ChromaDB embeddings
├── minerals.csv             # Local CSV dataset (Optional; loaded directly if present)
├── app.py                   # Main Python CLI entrypoint execution script
├── app_ui.py                # Streamlit Web UI chat application
├── README.md                # System technical documentation and workflow specifications
└── requirements.txt         # Declared python dependencies version sheet
```

---

## Configuration & Parameters

The system behavior can be tuned by modifying parameters globally inside the runtime execution file.

| Parameter | Default Value | Target Component | Purpose |
| :--- | :--- | :--- | :--- |
| `CHROMA_DB_DIR` | `./chroma_db` | Vector Store | Target directory for persisting vector embeddings on disk. |
| `EMBEDDING_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFaceEmbeddings | Multilingual sentence transformer model for dense vector generation. |
| `LOCAL_CSV_PATH` | `minerals.csv` | Data Ingestion | Path to local CSV file to prioritize over Kaggle download. |
| `CORE_TEXT_COLS` | `['Name', 'Crystal Structure', ...]` | Document Construction | Core physical/optical properties embedded into `page_content`. |
| `df.iloc[:, 9:135]` | Columns J to EE | Data Ingestion | Exact column indices for extracting chemical composition data (> 0). |
| `model` | `phi3` | ChatOllama | Target local LLM backend optimized for 4GB VRAM. |
| `search_kwargs`| `{"k": 3}` | Chroma VectorDB | Number of top relevant document snippets retrieved per query. |
| `temperature`  | `0` | ChatOllama LLM | Set to zero to eliminate creative hallucinations and enforce deterministic output. |

---

## Implementation Code Snippet

Below is the Streamlit Web UI application logic in `app_ui.py`:

```python
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

@st.cache_resource(show_spinner="Initializing vector store...")
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        try:
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
            _ = vectorstore.similarity_search("test", k=1)
            return vectorstore
        except Exception:
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)

    df = pd.read_csv(LOCAL_CSV_PATH) if os.path.exists(LOCAL_CSV_PATH) else pd.read_csv("minerals.csv")
    dynamic_chem_cols = df.iloc[:, 9:135].columns.tolist()
    # Document building logic...

@st.cache_resource(show_spinner="Initializing LLM chain...")
def get_rag_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="phi3", temperature=0)
    # RAG Chain setup with bullet point rules...

def main():
    st.set_page_config(page_title="Mineral Database RAG", page_icon="💎", layout="wide")
    st.title("💎 Mineral Database Local RAG System")

    rag_chain = get_rag_chain()
    # Chat interaction with real-time st.write_stream streaming...
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

2. **Vector Dimension Mismatch Error**
   - **Cause:** Switching embedding models when `./chroma_db` contains old vector indexes.
   - **Solution:** The application automatically detects dimension mismatches, clears `./chroma_db`, and rebuilds vector indexes seamlessly.

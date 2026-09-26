# Local RAG System for Mineral Database

A lightweight local Retrieval-Augmented Generation (RAG) system for querying mineralogical data offline.

## Models & Technical Specifications

* **LLM Model**: Ollama `phi3` (Microsoft Phi-3 Mini 3.8B, running locally)
* **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace Multilingual Sentence Transformer)
* **Vector Store**: Chroma DB (Cosine similarity space: `hnsw:space=cosine`)
* **Core Frameworks**: LangChain, Pandas, Streamlit, `thefuzz` (fuzzy string matching)

## Setup & Installation

**1. Prerequisites**
* Python 3.10 or higher.
* [Ollama](https://ollama.com/) installed on your machine.

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```
*(Alternatively: `pip install pandas langchain langchain-community langchain-chroma langchain-huggingface langchain-ollama sentence-transformers streamlit thefuzz`)*

**3. Configure Local LLM**
Ensure the Ollama service is running, then pull the required model:
```bash
ollama serve
ollama pull phi3
```

**4. First Launch & Data Ingestion**
Run either the Streamlit Web UI or the Terminal CLI. 
*Note: On the first launch, the system will automatically parse `minerals.csv`, generate embeddings, and build the `./chroma_db` cache. Subsequent launches will load instantly.*
```bash
# Launch the modern Web UI
streamlit run app_ui.py

# OR launch the Terminal Debugger
python app_cli.py
```

## Simplified System Workflow

```mermaid
graph TD
    Start([User Input Query]) --> DirectMatch{Direct Match Bypass: <br/> Exact Match in CSV?}

    %% Direct Match Path
    DirectMatch -- Yes --> Stage3[Stage 3: Filter Chroma DB <br/> filter='Name': target_mineral]

    %% Stage 1 & 2 Path
    DirectMatch -- No --> Stage1[Stage 1: translator_chain <br/> Extract Formal English Name]
    Stage1 --> Stage2{Stage 2: Gatekeeper Check <br/> Fuzzy/Regex Valid Name?}

    %% Failure Path
    Stage2 -- No --> LogMissing[Log query to missing_minerals.txt]
    LogMissing --> Reject([Output REJECTION_MESSAGE <br/> Halt Execution])

    %% Success Path
    Stage2 -- Yes --> Stage3

    %% Stage 3 Check
    Stage3 --> DocsFound{Docs Found <br/> in Chroma DB?}
    DocsFound -- No --> LogMissing
    DocsFound -- Yes --> Stage4[Stage 4: strict_chain <br/> Checklist Requirement & Stop Generation]

    %% Final Output
    Stage4 --> Final([Output Grounded Answer])

    %% Styling
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef check fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef error fill:#ffebee,stroke:#d32f2f,stroke-width:2px;
    classDef endpoint fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px;

    class Stage1,Stage3,Stage4 process;
    class DirectMatch,Stage2,DocsFound check;
    class LogMissing,Reject error;
    class Start,Final endpoint;
```

## Project Directory Structure

```text
mineral-rag/
│
├── chroma_db/               # Local persistent storage directory for ChromaDB embeddings
├── minerals.csv             # Local CSV dataset
├── missing_minerals.txt     # Log file recording unmatched raw queries and extracted terms
├── mineral_rag.py           # Core logic module (embeddings, vectorstore, LLM chains, fuzzy matching)
├── app_cli.py               # Terminal debugger CLI interface script
├── app_ui.py                # Streamlit Web UI chat interface script
├── README.md                # System documentation
└── requirements.txt         # Declared Python dependencies
```

## Searchable Mineral Properties & Data Source

The mineralogy database (`minerals.csv`) used by this local RAG system is sourced from Kaggle's [Comprehensive Database of Minerals](https://www.kaggle.com/datasets/vinven7/comprehensive-database-of-minerals).

The system supports accurate vector retrieval and strict generation across the following core attributes:
- Crystal Structure
- Mohs Hardness
- Diaphaneity
- Specific Gravity
- Optical
- Refractive Index
- Dispersion
- Hydrated Water
- Molar Mass
- Molar Volume
- Calculated Density
- Chemical Composition

## Demo

Streamlit chat UI displaying laser-focused property extraction and zero-hallucination guardrails:

![Streamlit Web Interface](streamlit-demo.jpg)
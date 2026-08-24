# Agentic RAG System

An Agentic Retrieval-Augmented Generation (RAG) system that answers questions from a PDF document using semantic retrieval, an LLM-powered Researcher Agent, and a Reviewer Agent.

## Architecture

```text
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Local Embeddings
 ↓
Qdrant Vector Database
 ↓
Retriever
 ↓
Researcher Agent
 ↓
OpenRouter LLM
 ↓
Reviewer Agent
 ↓
Final Answer
```

## Features

- PDF ingestion and text extraction
- Text chunking
- Local semantic embeddings
- Qdrant vector search
- Researcher Agent
- Reviewer Agent
- LangGraph workflow
- OpenRouter LLM integration
- Source and page tracking
- Execution logs
- Streamlit frontend
- Test cases

## Technologies

- Python
- PyMuPDF
- LangChain
- LangGraph
- Sentence Transformers
- Qdrant
- OpenRouter
- Streamlit

## Embeddings

The project uses the local embedding model:

```
BAAI/bge-small-en-v1.5
```

Embeddings are generated locally to avoid embedding API costs and rate limits.

## LLM

The LLM is accessed through OpenRouter using an OpenAI-compatible API.

Example:

```python
llm = ChatOpenAI(
    model="openrouter/free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)
```

## Project Structure

```
agentic-rag-system/
│
├── Rag/
│   ├── ingestion.py
│   └── retrieval.py
│
├── agents/
│   ├── researcher.py
│   └── reviewer.py
│
├── graph/
│   └── workflow.py
│
├── logs/
│   └── logger.py
│
├── tests/
│   ├── test_questions.json
│   └── run_tests.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── book.pdf
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

Add the PDF to:

```
data/book.pdf
```

## Run

1. **Ingestion**

```bash
python Rag/ingestion.py
```

2. **Test Retrieval**

```bash
python Rag/retrieval.py
```

3. **Test Researcher**

```bash
python agents/researcher.py
```

4. **Run Agentic Workflow**

```bash
python graph/workflow.py
```

5. **Run Frontend**

```bash
streamlit run app/streamlit_app.py
```

## RAG Workflow

The system first retrieves relevant chunks from Qdrant. The Researcher Agent generates an answer using only the retrieved context. The Reviewer Agent then checks whether the answer is supported by the retrieved evidence.

If the answer is not sufficiently supported, it is marked as:

```
UNSUPPORTED
```

Otherwise:

```
SUPPORTED
```

## Logging

Execution results are stored in:

```
logs/results.jsonl
```

The logs contain the question, retrieved sources, generated answer, and reviewer result.

## Security

API keys should never be committed to GitHub.
Add `.env` to `.gitignore`.

## Project Goal

The goal of this project is to demonstrate a complete Agentic RAG pipeline combining document retrieval, vector search, LLM generation, agent-based verification, logging, and a simple frontend.

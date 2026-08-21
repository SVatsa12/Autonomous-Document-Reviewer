# Universal Compliance Pipeline

This project implements a domain-aware document compliance system that can ingest multiple document types, extract clauses, build a vector index, run configurable compliance checks, and optionally answer natural-language questions against the indexed content.

It is no longer limited to a single rental-contract workflow. The current codebase supports:

- PDF, DOCX, and image-based documents
- Auto-detection of document domain (legal, banking, HR, generic)
- Custom rules loaded from JSON configuration files
- Compliance scoring and rule pass/fail reporting
- Semantic/vector search and RAG-style questions
- Offline fallback behavior when LLM APIs are unavailable or quota-limited

---

## What the project does

The system processes a document in this flow:

1. Load and validate the input file
2. Extract text from PDF, DOCX, or image inputs
3. Split the content into chunks
4. Extract clauses using LLM logic or local fallback logic
5. Build a FAISS-backed vector database for semantic retrieval
6. Evaluate compliance rules against the extracted clauses
7. Optionally answer questions using the indexed clauses/regulations

This is implemented across the core modules:

- `app.py` — command-line entry point
- `configurable_pipeline.py` — end-to-end processing pipeline
- `document_processor.py` — multi-format document ingestion
- `compliance_engine.py` — rules and rule evaluation engine
- `vector_db.py` — FAISS vector database + persistence
- `rag_engine.py` — RAG query layer
- `query_contract.py` — query CLI
- `offline_query.py` — offline fallback querying

---

## Supported inputs

The current `DocumentProcessor` supports these file formats:

- `.pdf`
- `.docx`
- `.png`
- `.jpg` / `.jpeg`
- `.tiff` / `.tif`

It also includes OCR fallback support for scanned images / scanned PDFs when the required libraries are installed.

---

## Supported domains

The app can target or auto-detect these domains:

- `legal`
- `banking`
- `hr`
- `generic`
- `insurance` (template generation supported)
- `real_estate` (template generation supported)

Domain detection is based on filename and content heuristics in `document_processor.py`.

---

## Environment setup

1. Create a virtual environment if needed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key(s) in a local `.env` file or environment variables.

Common variables used by the project:

- `GROQ_API_KEY` — used by the Groq-powered pipeline and query tools
- `GOOGLE_API_KEY` — used by Google GenAI-based embedding/extraction flows when applicable

Example `.env`:

```bash
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

The loader in `env_load.py` will try to load `.env` automatically when present.

> Do not commit real API keys to source control.

---

## Quick start

### Run the universal pipeline

```bash
python app.py --file path/to/document.pdf
```

### Specify a domain and rules file

```bash
python app.py --file loan_application.docx --domain banking --rules rules_banking.json
```

### Show file information without processing

```bash
python app.py --info policy.docx
```

### Create a rules template for a new domain

```bash
python app.py --create-rules insurance
```

### Skip compliance checks

```bash
python app.py --file contract.pdf --no-compliance
```

### Disable RAG features

```bash
python app.py --file contract.pdf --no-rag
```

---

## CLI options

`app.py` supports the following options:

```bash
python app.py --file <path> [--domain legal|banking|hr|generic|insurance|real_estate]
              [--rules <path>]
              [--no-compliance]
              [--no-rag]
              [--output <path>]
              [--create-rules DOMAIN]
              [--info]
```

The app writes JSON results to the selected output file, defaulting to `results.json`.

---

## Rule engine

The compliance engine supports multiple rule types:

- `threshold` — max/min/range/ratio checks
- `presence` — required clause/concept checks
- `forbidden` — blocked language or prohibited terms
- `semantic` — semantic comparison using vector/search context and LLM reasoning

These rules can be loaded from rule JSON files such as:

- `rules_legal.json`
- `rules_banking.json`
- `rules_hr.json`

The engine evaluates rule results and produces:

- pass/fail state
- severity level
- evidence and recommendations
- compliance score and overall status
- critical issue summaries

---

## RAG and vector search

The project includes a retrieval layer for asking natural-language questions against the processed document and supporting regulations.

### Query examples

```bash
python query_contract.py -q "What is the security deposit amount?"
```

Interactive mode:

```bash
python query_contract.py
```

Offline fallback:

```bash
python offline_query.py
```

The vector database is backed by FAISS and supports:

- clause indexing
- regulation metadata search
- hybrid retrieval (semantic + lexical ranking)
- persistence to disk
- offline fallback behavior when LLM calls fail

---

## Legacy pipeline

The repository still contains a legacy, contract-focused flow in `main.py` for direct processing of rental/lease-style documents:

```bash
python main.py
```

This older path is useful for contract/rental analysis and writes clause artifacts, but the universal flow in `app.py` is the primary current interface.

---

## Key files

| File | Purpose |
|---|---|
| `app.py` | Unified CLI entry point |
| `configurable_pipeline.py` | End-to-end processing pipeline |
| `document_processor.py` | Multi-format document ingestion and domain detection |
| `compliance_engine.py` | Rule engine and compliance evaluation |
| `vector_db.py` | FAISS vector database and persistence |
| `rag_engine.py` | Retrieval and answer generation |
| `query_contract.py` | Command-line RAG interface |
| `offline_query.py` | Offline querying fallback |
| `rules_*.json` | Domain rule definitions |
| `regulations.json` | Regulatory corpus used by retrieval |

---

## Outputs

The app produces structured JSON output such as:

- `results.json` — main pipeline output
- `clause_vectors.json` — vector store persistence
- `clause_vectors_<domain>.json` — domain-specific vector store path used in some runs
- `compliance.json` — compliance output in some workflows

The pipeline result includes:

- extracted clauses
- domain metadata
- vector DB path
- compliance score and rule results
- stats such as characters extracted and chunks processed

---

## Dependencies

The project uses packages including:

- `google-genai`
- `groq`
- `pypdf`
- `python-docx`
- `python-dotenv`
- `faiss-cpu`
- `numpy`
- `pillow`
- `pytesseract`
- `pdf2image`

Install them with:

```bash
pip install -r requirements.txt
```

---

## Notes

- The project is designed to work even when API access is limited; many components degrade gracefully.
- RAG behavior can run in offline mode when the LLM client is unavailable or quota is exceeded.
- For new use cases, create a domain rule file with `--create-rules DOMAIN` and customize the generated JSON.

---

## Recommended workflow

For practical use, the recommended path is:

```bash
python app.py --file path/to/document.pdf --domain legal
```

Then, if you want to query the generated clause database:

```bash
python query_contract.py -q "Summarize the key obligations in this document"
```

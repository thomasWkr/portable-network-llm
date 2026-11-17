# Offline document retrieval

This pipeline indexes the user-reviewed PDFs directly. It does not run the downloader's relevance checks, delete documents, or require an approved manifest. Original PDFs remain unchanged. Exact duplicates were removed once, with a record at `data/rag/duplicate_files.json`; the download report points to the retained copies.

## Setup and rebuild

Run from the repository root. Poppler's `pdftotext` must be installed (on Debian/Ubuntu: `sudo apt install poppler-utils`).

```bash
python -m venv .venv-rag
.venv-rag/bin/pip install -r requirements/requirements-rag.txt
.venv-rag/bin/python prepare_rag_model.py
.venv-rag/bin/python rag_pipeline.py build
```

Only model preparation and dependency installation require internet. The model download is pinned to an official Hugging Face revision. Extraction, embeddings, and queries run locally without remote embedding services or a GPU. Model preparation downloads the ONNX encoder and tokenizer; it does not download any of the five generative LLMs.

Default input collections:

- `input_sources/manuals/`: equipment documents.
- `input_sources/other_sources/`: general networking/PON sources.

Both folders are read recursively. All remaining PDFs are included; textless pages are reported, not deleted. If exact duplicate PDFs are added later, building fails explicitly rather than silently overwriting their source records.

## Extraction and passage creation

Poppler layout extraction preserves line breaks and horizontal column spacing, including textual tables. This is a text-first index: image-only diagrams, scanned text and formula layout are not interpreted. No OCR is applied. `extraction_report.json` lists extraction failures and empty/low-text pages, which may be figures or genuinely blank pages.

Cleaning normalizes Unicode and removes common repeated margin boilerplate and page numbers from extracted text only. Commands and technical values are preserved. Passages stay within a physical PDF page and prefer paragraph boundaries, then line boundaries for long tables/procedures. Long passages use tokenizer offsets, at most 350 body tokens and 50 tokens of overlap. These parameters can be changed with `--chunk-tokens` and `--overlap`. Physical PDF page numbers start at 1; printed page labels may differ.

The embedding model is `intfloat/multilingual-e5-small`: 384-dimensional float32 vectors, masked-mean pooling and L2 normalization. Passage inputs use `passage: ` and question inputs use `query: `, following the model's documentation. The document filename supplies a short title context. Inputs exceeding the 512-token model limit fail rather than being silently truncated. The same encoder/tokenizer fingerprint must be used for indexing and searching.

Manufacturer/model associations come from the download report, supplemented by exact filename matches against `input_sources/onts_onus_olts_roteadores_filtrados.csv`. Deleted duplicate filenames are also considered, preserving aliases for manuals covering multiple models. This is technical source metadata, not a new document approval step. Unmatched PDFs remain searchable, but cannot be selected by an equipment filter until their association is supplied.

## Query examples

```bash
.venv-rag/bin/python rag_pipeline.py query 'Qual é a função da ONU em uma rede GPON?'

.venv-rag/bin/python rag_pipeline.py query \
  'Como configurar uma VLAN?' --source equipment --manufacturer Intelbras

.venv-rag/bin/python rag_pipeline.py query \
  'Como configurar o equipamento?' --model HG6145F --manufacturer "Wuhan FiberHome"

.venv-rag/bin/python rag_pipeline.py query \
  'Como funciona o orçamento de potência óptica?' --source network --json
```

Manufacturer selection is an accent/punctuation-insensitive prefix of the legal manufacturer name, so use `Wuhan FiberHome` for a manufacturer stored as `Wuhan FiberHome International ...`. Model selection requires an exact normalized identifier, preventing AX100 from matching AX1000. `--include-network` adds general sources when using equipment filters. `--top-k` defaults to 5.

Retrieval merges cosine-similarity ranking and SQLite FTS5/BM25 keyword ranking using reciprocal rank fusion. Keyword retrieval removes common Portuguese/English question words and, after equipment selection, the selected equipment name so it focuses on the technical issue. It is useful for identifiers and commands; multilingual vectors allow Portuguese questions to find English passages. Metadata filters apply to both rankings. Up to two returned passages may come from the same document page. Similarity and rank scores are not calibrated confidence probabilities.

Output consists of evidence passages with document ID, source path, collection, equipment aliases, and physical page. `--json` provides machine-readable results for a later LLM prompt or benchmark harness. This command retrieves evidence; it does not generate answers or verify that a question is answerable.

## Outputs and interrupted builds

All generated files are under `data/rag/` and ignored by Git:

| File | Purpose |
|---|---|
| `index.sqlite` | Documents, equipment aliases, passages and FTS5 keyword index |
| `embeddings.npy` | Normalized vectors; row 0 matches chunk rowid 1 |
| `model/` | Local ONNX query encoder, tokenizer and pinned model settings |
| `build_info.json` | Completed-build dimensions, counts, configuration and checksums |
| `extraction_report.json` | Extraction coverage and pages needing inspection |
| `extracted/` | Reusable page text keyed by PDF SHA-256 |
| `embedding_cache.sqlite` | Reusable vectors keyed by encoder and passage content |
| `duplicate_files.json` | One-time deletion record and retained-file mappings |

Rerun `build` after interruptions. Extraction and committed vector batches are cached; source changes and changed passage text get new cache keys. Builds use staging files and publish `build_info.json` last. Query checksums reject incomplete or mismatched output. Do not run two builds into the same output folder concurrently. Default embedding batch size is 8; `--threads` is 4 and `--extract-workers` is 4.

For a later Android implementation, retain `index.sqlite`, `embeddings.npy`, `build_info.json` and the query encoder/tokenizer. The ONNX model can be used with an Android ONNX runtime; the Python CLI is the desktop prototype. The SQLite FTS5 extension must be available in the chosen Android SQLite library. PDF source paths are repository-relative and need mapping to the app's local document storage for citation viewing. The query-only generation controller and phone prototype are described in [MOBILE.md](MOBILE.md).

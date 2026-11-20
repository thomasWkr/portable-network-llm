# portable-network-llm

O [guia de teste completo no celular](docs/MOBILE.md#teste-completo-no-celular) descreve o empacotamento, a execução com os conjuntos `questions_broad.csv` e `questions_specific.csv` e a coleta dos resultados.

Download equipment manuals from the filtered CSV:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements/requirements-manuals.txt
.venv/bin/python download_manuals.py --limit 5
.venv/bin/python download_manuals.py
```

The downloader tries model-only and brand/model searches, direct PDFs, download/support pages, and embedded PDF viewers. It follows up to two link levels and 30 links per result (`--max-depth`, `--max-links`). Exact model boundaries and networking context are required; unrelated camera, phone, and tool documents are rejected. Brand and manual wording are recorded as evidence but are optional by default, allowing OEM documents and technical specifications. Use `--strict-validation` to require both. Scanned PDFs without searchable text remain unsupported. All downloads are candidates; review hardware revisions before use.

Manuals must be available over HTTPS without sign-in. The downloader checks each candidate and redirect against `OFFICIAL_MANUAL_DOMAINS` in `download_manuals.py`. This list comes from the corpus source review. Add a site's domain after verifying that the manufacturer or product brand owns it. Source verification is independent of copyright and confidentiality notices.

Defaults follow the new layout:

- Catalog: `input_sources/onts_onus_olts_roteadores_filtrados.csv`
- PDFs: `input_sources/manuals/`
- Resume report: `input_sources/relatorio_download_manuais.csv`
- Search log: `data/manuals_search.log`
- Additional RAG sources: `input_sources/other_sources/`

Run `python listCpes.py` to filter `input_sources/Produtos_Homologados_Anatel.csv`. Downloader paths can be overridden with `--input`, `--output`, `--report`, and `--log-file`. Run commands from the repository root.

```bash
.venv/bin/python download_manuals.py --retry-failed --workers 3 --delay 3 --limit 5
```

Successful files resume by checksum. Moved PDFs are recovered by filename in the selected output directory. Older validation versions are checked against the official public URL before reuse. Use `--revalidate` to force a local check on files already verified under the current rules. Rejected files are preserved for review while replacement searches run. Previous unsuccessful rows require `--retry-failed`. Reports save atomically after every completed device.

DDGS defaults to the ordered engine list `yahoo,duckduckgo,brave,startpage,mojeek`, trying Yahoo first. Each engine is tried separately until one returns results. Transport failures cool that engine down for 120 seconds across workers, then allow a later probe. Three consecutive empty responses trigger a 60-second cooldown because DDGS can hide rate limits as empty results. Unavailable engine names are rejected before DDGS can silently switch to automatic engines. Override with `--backends` to test engines supported by your installed DDGS version. Exhausting every query with transport failures produces `search_error`; completed searches without a matching PDF produce `not_found`.

Search starts are serialized and spaced by `--delay` (default 1 second). Three workers download concurrently by default. `--search-timeout` defaults to 8 seconds per engine request; `--results` defaults to 6 per query and `--max-mb` to 40 per response. Queries, engine failures/cooldowns, fetches, and validation evidence are logged. Search results, downloads, and extracted text are cached across workers during each run.

For the official Brave API, set `BRAVE_API_KEY` and use `--provider brave`. This is separate from the DDGS Brave backend and may consume paid API quota; no automatic switch occurs.

The offline embedding and retrieval pipeline is documented in [docs/RAG.md](docs/RAG.md). It indexes the existing equipment PDFs and `input_sources/other_sources/`, with multilingual E5 vectors, SQLite keyword retrieval, and page references. Setup uses `requirements/requirements-rag.txt`; the entry points are `prepare_rag_model.py` and `rag_pipeline.py`.

The query-only LLM controller and Android prototype are described in [docs/MOBILE.md](docs/MOBILE.md). Start with `.venv-rag/bin/python llm_controller.py ask "Qual é a função da ONU?"`. The controller generates answers with page citations; batch runs capture raw answers and metrics for later evaluation.

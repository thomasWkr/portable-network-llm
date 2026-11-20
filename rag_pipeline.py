"""Build and search a local, multilingual PDF retrieval index (no LLM required)."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unicodedata

import numpy as np

LOG = logging.getLogger('rag')
FORMAT_VERSION = 1


def digest_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def normalized(value):
    text = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode().casefold()
    return re.sub(r'[^a-z0-9]', '', text)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class E5Encoder:
    """Official ONNX E5: masked mean pooling followed by L2 normalization."""
    def __init__(self, model_dir='data/rag/model', threads=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.folder = Path(model_dir)
        self.info = json.loads((self.folder / 'embedding_model.json').read_text())
        self.tokenizer = Tokenizer.from_file(str(self.folder / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.folder / self.info['onnx_file']),
                                           sess_options=options, providers=['CPUExecutionProvider'])
        self.model_hash = digest_file(self.folder / self.info['onnx_file'])
        self.fingerprint = hashlib.sha256((self.model_hash + digest_file(self.folder / 'tokenizer.json')
                                          + 'masked-mean-l2-v1').encode()).hexdigest()
        LOG.info('Loaded %s (%s), CPU threads=%s', self.info['model_id'], self.info['revision'], threads)

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def encode(self, texts):
        encoded = self.tokenizer.encode_batch(texts)
        length = max(len(item.ids) for item in encoded)
        if length > 512:
            raise ValueError(f'Embedding input has {length} tokens; refusing silent truncation')
        size = len(encoded)
        ids = np.full((size, length), self.tokenizer.token_to_id('<pad>'), dtype=np.int64)
        mask = np.zeros_like(ids)
        for index, item in enumerate(encoded):
            ids[index, :len(item.ids)] = item.ids
            mask[index, :len(item.ids)] = 1
        inputs = {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': np.zeros_like(ids)}
        feed = {item.name: inputs[item.name] for item in self.session.get_inputs()}
        hidden = self.session.run(None, feed)[0]
        if hidden.ndim != 3:
            raise ValueError(f'Expected token embeddings, got {hidden.shape}')
        weights = mask[..., None].astype(np.float32)
        vectors = (hidden * weights).sum(axis=1) / weights.sum(axis=1).clip(1)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
        if vectors.shape[1] != 384 or not np.isfinite(vectors).all():
            raise ValueError('Invalid embeddings')
        return vectors.astype(np.float32)


def report_equipment(report):
    associations = {}
    if not Path(report).exists():
        return associations
    with Path(report).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if not row.get('path'):
                continue
            source = Path(row['path'])
            moved = Path(report).parent / 'manuals' / source.name
            if not source.is_file() and moved.is_file():
                source = moved
            key = str(source.resolve())
            equipment = {'manufacturer': row.get('Nome do Fabricante', ''), 'model': row.get('Modelo', '')}
            if equipment not in associations.setdefault(key, []):
                associations[key].append(equipment)
    return associations


def discover(roots, report):
    associations = report_equipment(report)
    documents = {}
    for collection, root in roots:
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f'Source folder missing: {root}')
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.casefold() != '.pdf':
                continue
            sha = digest_file(path)
            equipment = associations.get(str(path.resolve()), [])
            if sha in documents:
                document = documents[sha]
                document['duplicate_paths'][str(path)] = collection
                for item in equipment:
                    if item not in document['equipment']:
                        document['equipment'].append(item)
                LOG.info('Skipping duplicate PDF %s (using %s)', path, document['path'])
                continue
            documents[sha] = {'id': sha[:24], 'sha256': sha, 'path': str(path),
                              'collection': collection, 'title': path.stem,
                              'equipment': list(equipment), 'duplicate_paths': {}}
    if not documents:
        raise ValueError('No PDFs found')
    return list(documents.values())


def add_catalog_associations(documents, catalog_path, duplicate_path):
    """Recover equipment aliases from retained, skipped, and previously removed filenames."""
    if not Path(catalog_path).exists():
        return
    filenames = {}
    by_path = {}
    for doc in documents:
        for path, collection in [(doc['path'], doc['collection']), *doc.get('duplicate_paths', {}).items()]:
            by_path[str(Path(path).resolve())] = doc
            if collection == 'equipment':
                filenames[Path(path).name] = doc
    if Path(duplicate_path).exists():
        by_hash = {doc.get('sha256'): doc for doc in documents if doc.get('sha256')}
        for item in json.loads(Path(duplicate_path).read_text()):
            digest = item.get('sha256')
            doc = by_hash.get(digest) if digest else by_path.get(str(Path(item['kept']).resolve()))
            if doc:
                filenames[Path(item['removed']).name] = doc
    with Path(catalog_path).open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            maker, model = row.get('Nome do Fabricante', ''), row.get('Modelo', '')
            if not maker or not model:
                continue
            stem = re.sub(r'[^\w.-]+', '_', f'{maker}_{model}', flags=re.UNICODE)
            patterns = [re.compile(re.escape(name) + r'(?:_[0-9a-f]{8})?(?:_[0-9a-f]{12})?\.pdf$')
                        for name in {stem, stem[:120]}]
            for filename, doc in filenames.items():
                if any(pattern.fullmatch(filename) for pattern in patterns):
                    item = {'manufacturer': maker, 'model': model}
                    if item not in doc['equipment']:
                        doc['equipment'].append(item)


def remove_duplicate_files(documents, report, duplicate_path):
    removals = []
    for doc in documents:
        kept = Path(doc['path']).resolve()
        for source in doc['duplicate_paths']:
            path = Path(source)
            if path.resolve() == kept or not path.is_file() or digest_file(path) != doc['sha256']:
                continue
            removals.append({'removed': source, 'kept': doc['path'], 'sha256': doc['sha256']})
    if not removals:
        return 0

    # Save the aliases and redirect download records before unlinking any copies.
    history = json.loads(duplicate_path.read_text()) if duplicate_path.exists() else []
    history.extend(item for item in removals if item not in history)
    write_json(duplicate_path, history)
    report = Path(report)
    if report.is_file():
        redirects = {Path(item['removed']).resolve(): item for item in removals}
        with report.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = reader.fieldnames
        changed = False
        for row in rows:
            item = redirects.get(Path(row['path']).resolve()) if row.get('path') else None
            if item and row.get('sha256') in (None, '', item['sha256']):
                row['path'] = str(Path(item['kept']).resolve())
                changed = True
        if changed:
            with tempfile.TemporaryDirectory(dir=report.parent) as folder:
                temporary = Path(folder) / report.name
                with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                temporary.replace(report)

    removed = 0
    for item in removals:
        path, kept = Path(item['removed']), Path(item['kept'])
        if not path.is_file() or not kept.is_file() or any(
                digest_file(copy) != item['sha256'] for copy in (path, kept)):
            LOG.warning('Duplicate changed before removal: %s', path)
            continue
        path.unlink()
        removed += 1
        LOG.info('Removed duplicate PDF %s (kept %s)', path, kept)
    return removed


def extract(document, cache):
    """Poppler layout extraction keeps line breaks and column/table spacing."""
    target = Path(cache) / (document['sha256'] + '.json')
    if target.exists():
        return json.loads(target.read_text(encoding='utf-8'))
    try:
        result = subprocess.run(['pdftotext', '-layout', '-enc', 'UTF-8', document['path'], '-'],
                                capture_output=True, timeout=240, check=True)
        content = result.stdout.decode('utf-8', errors='replace')
        pages = content.split('\f')
        if pages and not pages[-1].strip():
            pages.pop()
        entry = {'pages': pages, 'error': None, 'warnings': result.stderr.decode('utf-8', errors='replace')[:2000]}
    except (subprocess.SubprocessError, OSError) as exc:
        entry = {'pages': [], 'error': str(exc), 'warnings': ''}
    # Do not cache failures: the next run can retry a transient extraction error.
    if not entry['error']:
        write_json(target, entry)
    return entry


def clean_pages(pages):
    cleaned = [unicodedata.normalize('NFKC', page).replace('\u00ad', '').replace('\x00', '') for page in pages]
    # Remove only frequent page-margin lines; keep values, commands and tables.
    edges = Counter()
    for page in cleaned:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        edges.update(set(lines[:2] + lines[-2:]))
    repeated = {line for line, count in edges.items()
                if len(cleaned) >= 5 and count >= max(3, len(cleaned) * .7)
                and 4 <= len(line) <= 100
                and re.search(r'copyright|all rights reserved|confidential|www\.|https?://|©|user manual|installation guide', line, re.I)}
    output = []
    for page in cleaned:
        lines = page.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        margin_indices = set(nonempty[:2] + nonempty[-2:])
        lines = [line.rstrip() for i, line in enumerate(lines)
                 if not (i in margin_indices and (line.strip() in repeated
                         or re.fullmatch(r'(?:page\s+)?\d+(?:\s+of\s+\d+)?', line.strip(), re.I)))]
        output.append('\n'.join(lines).strip())
    return output


def page_chunks(text, tokenizer, max_tokens=350, overlap=50):
    """Prefer paragraph boundaries, with token-offset windows for long sections."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    offsets = encoded.offsets
    if not offsets:
        return
    paragraphs = {match.end() for match in re.finditer(r'\n\s*\n', text)}
    line_breaks = {match.end() for match in re.finditer(r'\n', text)}
    start = 0
    while start < len(offsets):
        end = min(start + max_tokens, len(offsets))
        if end < len(offsets):
            candidates = [i for i in range(start + max_tokens * 3 // 4, end)
                          if any(offsets[i - 1][1] <= boundary <= offsets[i][0] for boundary in paragraphs)]
            if not candidates:
                candidates = [i for i in range(start + max_tokens * 3 // 4, end)
                              if any(offsets[i - 1][1] <= boundary <= offsets[i][0]
                                     for boundary in line_breaks)]
            if candidates:
                end = candidates[-1]
        body = text[offsets[start][0]:offsets[end - 1][1]].strip()
        if body:
            yield body
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap)


def create_database(path):
    db = sqlite3.connect(path)
    db.executescript('''
        CREATE TABLE documents(id TEXT PRIMARY KEY, sha256 TEXT, path TEXT, collection TEXT, title TEXT, pages INTEGER);
        CREATE TABLE equipment(document_id TEXT, manufacturer TEXT, model TEXT, manufacturer_norm TEXT, model_norm TEXT);
        CREATE INDEX equipment_model ON equipment(model_norm);
        CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, id TEXT UNIQUE, document_id TEXT,
                            page INTEGER, section TEXT, text TEXT, embedding_text TEXT, token_count INTEGER);
        CREATE INDEX chunks_document ON chunks(document_id);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='rowid',
                                                  tokenize='unicode61 remove_diacritics 2');
    ''')
    return db


def build(args):
    if not shutil.which('pdftotext'):
        raise SystemExit('pdftotext is required (install poppler-utils)')
    started = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    encoder = E5Encoder(args.model_dir, args.threads)
    documents = discover([('equipment', args.equipment), ('network', args.network)], args.report)
    duplicate_path = output / 'duplicate_files.json'
    add_catalog_associations(documents, args.equipment_csv, duplicate_path)
    staging = output / 'index.sqlite.tmp'
    if staging.exists():
        staging.unlink()
    db = create_database(staging)
    issues, blank, low_text, total_pages = [], [], [], 0
    cache = output / 'extracted'
    cache.mkdir(exist_ok=True)
    extraction_rows = []
    with ThreadPoolExecutor(max_workers=args.extract_workers) as pool:
        extracted = pool.map(lambda doc: extract(doc, cache), documents)
        for number, (doc, entry) in enumerate(zip(documents, extracted), 1):
            pages = clean_pages(entry['pages'])
            doc_chunks = 0
            db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?)',
                       (doc['id'], doc['sha256'], doc['path'], doc['collection'], doc['title'], len(pages)))
            for item in doc['equipment']:
                db.execute('INSERT INTO equipment VALUES(?,?,?,?,?)',
                           (doc['id'], item['manufacturer'], item['model'],
                            normalized(item['manufacturer']), normalized(item['model'])))
            if entry['error']:
                issues.append({'path': doc['path'], 'error': entry['error']})
            for page_number, page in enumerate(pages, 1):
                total_pages += 1
                if not page.strip():
                    blank.append({'path': doc['path'], 'page': page_number})
                    continue
                if sum(char.isalnum() for char in page) < 30:
                    low_text.append({'path': doc['path'], 'page': page_number})
                # Title context helps cross-document retrieval; never truncate an input silently.
                prefix = 'passage: ' + doc['title'][:100] + '\n'
                for chunk_number, body in enumerate(page_chunks(page, encoder.tokenizer,
                                                               args.chunk_tokens, args.overlap)):
                    embedding_text = prefix + body
                    token_count = len(encoder.tokenizer.encode(embedding_text).ids)
                    if token_count > 512:
                        raise ValueError(f'Chunk exceeds E5 limit: {doc["path"]} page {page_number}')
                    chunk_id = f'{doc["id"]}:p{page_number}:c{chunk_number}'
                    section = body.splitlines()[0].strip()[:160]
                    cursor = db.execute('INSERT INTO chunks(id,document_id,page,section,text,embedding_text,token_count) '
                                        'VALUES(?,?,?,?,?,?,?)',
                                        (chunk_id, doc['id'], page_number, section, body, embedding_text, token_count))
                    db.execute('INSERT INTO chunks_fts(rowid,text) VALUES(?,?)', (cursor.lastrowid, body))
                    doc_chunks += 1
            extraction_rows.append({'path': doc['path'], 'collection': doc['collection'],
                                    'pages': len(pages), 'chunks': doc_chunks, 'warning': entry['warnings']})
            LOG.info('EXTRACT [%s/%s] %s: %s pages, %s chunks', number, len(documents),
                     doc['path'], len(pages), doc_chunks)
    db.commit()
    extraction_report = {'documents': extraction_rows, 'errors': issues, 'empty_pages': blank,
                         'low_text_pages': low_text, 'note': 'No OCR; empty pages can be images or genuinely blank.'}
    write_json(output / 'extraction_report.json', extraction_report)
    count = db.execute('SELECT count(*) FROM chunks').fetchone()[0]
    if not count:
        raise ValueError('No extracted text to embed')
    vectors = np.lib.format.open_memmap(output / 'embeddings.npy.tmp', mode='w+', dtype=np.float32,
                                       shape=(count, 384))
    vector_cache = sqlite3.connect(output / 'embedding_cache.sqlite')
    vector_cache.execute('CREATE TABLE IF NOT EXISTS vectors(key TEXT PRIMARY KEY, vector BLOB)')
    rows = db.execute('SELECT rowid, embedding_text, token_count FROM chunks ORDER BY token_count, rowid').fetchall()
    computed, reused = 0, 0
    for begin in range(0, count, args.batch_size):
        batch = rows[begin:begin + args.batch_size]
        pending = []
        for rowid, text, tokens in batch:
            key = hashlib.sha256((encoder.fingerprint + text).encode()).hexdigest()
            hit = vector_cache.execute('SELECT vector FROM vectors WHERE key=?', (key,)).fetchone()
            if hit:
                vectors[rowid - 1] = np.frombuffer(hit[0], dtype=np.float32)
                reused += 1
            else:
                pending.append((rowid, text, key))
        if pending:
            values = encoder.encode([item[1] for item in pending])
            for (rowid, text, key), vector in zip(pending, values):
                vectors[rowid - 1] = vector
                vector_cache.execute('INSERT OR REPLACE INTO vectors VALUES(?,?)', (key, vector.tobytes()))
                computed += 1
        vector_cache.commit()
        if begin % (args.batch_size * 10) == 0 or begin + args.batch_size >= count:
            LOG.info('EMBED %s/%s; computed=%s cached=%s; elapsed %.1fs',
                     min(begin + args.batch_size, count), count, computed, reused, time.monotonic() - started)
    vectors.flush()
    del vectors
    vector_cache.close()
    db.close()
    for doc in documents:
        if not Path(doc['path']).is_file() or digest_file(doc['path']) != doc['sha256']:
            raise ValueError(f'Source PDF changed during the build: {doc["path"]}; rerun build')
    # The marker is written last. Queries verify the database/vector checksums.
    staging.replace(output / 'index.sqlite')
    (output / 'embeddings.npy.tmp').replace(output / 'embeddings.npy')
    info = {'format_version': FORMAT_VERSION, 'model': encoder.info, 'encoder_fingerprint': encoder.fingerprint,
            'documents': len(documents), 'pages': total_pages, 'chunks': count, 'dimensions': 384,
            'chunk_tokens': args.chunk_tokens, 'overlap': args.overlap,
            'extraction_errors': len(issues), 'empty_pages': len(blank), 'low_text_pages': len(low_text),
            'index_sha256': digest_file(output / 'index.sqlite'),
            'embeddings_sha256': digest_file(output / 'embeddings.npy'),
            'elapsed_seconds': round(time.monotonic() - started, 2)}
    write_json(output / 'build_info.json', info)
    LOG.info('Removed %s duplicate PDFs', remove_duplicate_files(documents, args.report, duplicate_path))
    print(json.dumps(info, ensure_ascii=False, indent=2))


def allowed_documents(db, source=None, model=None, manufacturer=None, include_network=False):
    clauses, params = [], []
    if source:
        clauses.append('d.collection=?')
        params.append(source)
    if model or manufacturer:
        constraints = ['e.document_id=d.id']
        if model:
            constraints.append('e.model_norm=?')
            params.append(normalized(model))
        if manufacturer:
            constraints.append('e.manufacturer_norm LIKE ?')
            params.append(normalized(manufacturer) + '%')
        condition = 'EXISTS(SELECT 1 FROM equipment e WHERE ' + ' AND '.join(constraints) + ')'
        if include_network:
            condition = '(' + condition + " OR d.collection='network')"
        clauses.append(condition)
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    return {row[0] for row in db.execute('SELECT d.id FROM documents d' + where, params)}


def keyword_expression(question, model=None, manufacturer=None):
    stopwords = set(('a an the and or of in on to for with is are do does how what which '
                     'can i my o a os as um uma uns umas de do da dos das em no na nos nas '
                     'que como qual quais para por e ou com sem ao aos e sao foi ser esta '
                     'estao meu minha se isso isto esse essa').split())
    selected = {normalized(word) for value in (model, manufacturer) if value
                for word in re.findall(r'\w+', value)}
    if model:
        selected.add(normalized(model))
    words = [word for word in dict.fromkeys(re.findall(r'\w+', question, flags=re.UNICODE))
             if normalized(word) not in stopwords | selected][:40]
    return ' OR '.join('"' + word.replace('"', '""') + '"' for word in words)


def search_index(db, vectors, query_vector, question, top_k=5, source=None, model=None,
                 manufacturer=None, include_network=False):
    allowed = allowed_documents(db, source, model, manufacturer, include_network)
    rows = db.execute('SELECT rowid, document_id FROM chunks').fetchall()
    candidates = np.asarray([rowid - 1 for rowid, docid in rows if docid in allowed], dtype=np.int64)
    if not len(candidates):
        return []
    scores = vectors[candidates] @ query_vector
    semantic_order = np.argsort(-scores, kind='stable')[:max(40, top_k * 8)]
    semantic = [int(candidates[index]) + 1 for index in semantic_order]
    similarity = {int(candidates[index]) + 1: float(scores[index]) for index in semantic_order}
    expression = keyword_expression(question, model, manufacturer)
    lexical = []
    if expression:
        # Rank after metadata filtering, rather than filtering an already limited global result set.
        lexical = [rowid for rowid, docid in db.execute(
            'SELECT f.rowid,c.document_id FROM chunks_fts f JOIN chunks c ON c.rowid=f.rowid '
            'WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts)', (expression,)) if docid in allowed][:max(40, top_k * 8)]
    rrf = {}
    for ranking in (semantic, lexical):
        for rank, rowid in enumerate(ranking, 1):
            rrf[rowid] = rrf.get(rowid, 0) + 1 / (60 + rank)
    output, per_page = [], Counter()
    for rowid in sorted(rrf, key=lambda key: (-rrf[key], key)):
        row = db.execute('SELECT c.id,c.page,c.section,c.text,d.id,d.path,d.collection,d.title '
                         'FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.rowid=?', (rowid,)).fetchone()
        chunk_id, page, section, text, docid, path, collection, title = row
        if per_page[(docid, page)] >= 2:
            continue
        per_page[(docid, page)] += 1
        equipment = [{'manufacturer': maker, 'model': name} for maker, name in db.execute(
            'SELECT manufacturer,model FROM equipment WHERE document_id=?', (docid,))]
        output.append({'chunk_id': chunk_id, 'document_id': docid, 'path': path,
                       'collection': collection, 'page': page, 'section': section, 'text': text,
                       'equipment': equipment, 'rrf_score': rrf[rowid],
                       'cosine_similarity': similarity.get(rowid, float(vectors[rowid - 1] @ query_vector))})
        if len(output) >= top_k:
            break
    return output


def query(args):
    output = Path(args.index)
    info = json.loads((output / 'build_info.json').read_text())
    for name, field in [('index.sqlite', 'index_sha256'), ('embeddings.npy', 'embeddings_sha256')]:
        if digest_file(output / name) != info[field]:
            raise ValueError(f'Index integrity check failed: {name}; rebuild the index')
    encoder = E5Encoder(args.model_dir, args.threads)
    if encoder.fingerprint != info['encoder_fingerprint']:
        raise ValueError('Query encoder differs from the index encoder')
    vector = encoder.encode(['query: ' + args.question])[0]
    vectors = np.load(output / 'embeddings.npy', mmap_mode='r')
    db = sqlite3.connect(f'file:{(output / "index.sqlite").resolve()}?mode=ro', uri=True)
    if len(vectors) != db.execute('SELECT count(*) FROM chunks').fetchone()[0]:
        raise ValueError('Vector/metadata count mismatch')
    results = search_index(db, vectors, vector, args.question, args.top_k, args.source,
                           args.model, args.manufacturer, args.include_network)
    db.close()
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for number, result in enumerate(results, 1):
            print(f'\n[{number}] {result["path"]} — PDF page {result["page"]}\n'
                  f'chunk={result["chunk_id"]} collection={result["collection"]}\n{result["text"]}')
        if not results:
            print('No matching documents; check metadata filters.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('build')
    builder.add_argument('--equipment', default='input_sources/manuals')
    builder.add_argument('--network', default='input_sources/other_sources')
    builder.add_argument('--equipment-csv', default='input_sources/onts_onus_olts_roteadores_filtrados.csv')
    builder.add_argument('--report', default='input_sources/relatorio_download_manuais.csv')
    builder.add_argument('--output', default='data/rag')
    builder.add_argument('--chunk-tokens', type=int, default=350)
    builder.add_argument('--overlap', type=int, default=50)
    builder.add_argument('--batch-size', type=int, default=8)
    builder.add_argument('--extract-workers', type=int, default=4)
    search = commands.add_parser('query')
    search.add_argument('question')
    search.add_argument('--index', default='data/rag')
    search.add_argument('--top-k', type=int, default=5)
    search.add_argument('--source', choices=['equipment', 'network'])
    search.add_argument('--model', help='Exact normalized model from download-report metadata')
    search.add_argument('--manufacturer', help='Manufacturer prefix from download-report metadata')
    search.add_argument('--include-network', action='store_true')
    search.add_argument('--json', action='store_true')
    for command in (builder, search):
        command.add_argument('--model-dir', default='data/rag/model')
        command.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.command == 'build':
        if not (1 <= args.overlap < args.chunk_tokens <= 400) or args.batch_size < 1 or args.extract_workers < 1:
            parser.error('Require 1 <= overlap < chunk-tokens <= 400 and positive worker/batch counts')
    elif args.top_k < 1:
        parser.error('--top-k must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    (build if args.command == 'build' else query)(args)


if __name__ == '__main__':
    main()

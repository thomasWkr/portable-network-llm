"""Discover and download candidate equipment manuals; resume using a CSV report."""
import argparse
import csv
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import io
import logging
import os
from pathlib import Path
import re
import time
import threading
import tempfile
import unicodedata
from urllib.parse import unquote, urljoin, urlsplit

import requests


MANUAL = re.compile(r'manual|user.?guide|installation.?guide|guia|instruções|datasheet', re.I)
LOG = logging.getLogger('manuals')
VALIDATION_VERSION = '3'
DEFAULT_BACKENDS = 'yahoo,duckduckgo,brave,startpage,mojeek'
FIELDS = ['Nome do Fabricante', 'Modelo', 'status', 'source_url', 'path', 'sha256', 'detail', 'validation_version']


def normalize(value):
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', value.lower())


def folded(value):
    return unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode().casefold()


def flexible_pattern(value):
    # Allow AX-100, AX 100 and AX100, but never AX1000 or unrelated substrings.
    parts = re.findall(r'[a-z]+|[0-9]+', folded(value))
    if not parts:
        return re.compile(r'(?!)')
    return re.compile(r'(?<![a-z0-9])' + r'[\s._/()-]*'.join(map(re.escape, parts))
                      + r'(?![a-z0-9])')


def manufacturer_names(maker):
    # Prefer the leading brand after optional location words; do not accept
    # generic words such as Technology, Networks or Electronics as identity.
    locations = {'shenzhen', 'zhuhai', 'guangzhou', 'wuhan', 'shanghai', 'beijing',
                 'hangzhou', 'dongguan', 'ningbo', 'suzhou', 'foshan', 'chengdu'}
    generic = {'co', 'company', 'corporation', 'corp', 'inc', 'ltd', 'limited',
               'ltda', 'sa', 's', 'a', 'industria', 'industries', 'industry',
               'technology', 'technologies', 'telecommunication', 'telecommunications',
               'telecomunicacao', 'telecomunicacoes', 'electronics', 'electronic',
               'eletronica', 'eletronicos', 'equipment', 'equipamentos', 'networks',
               'network', 'solutions', 'international', 'brasil', 'brazil', 'china',
               'de', 'do', 'da', 'e', 'and', 'the'}
    words = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', folded(maker))
    words = [word for word in words if word not in locations]
    brand = next((word for word in words if word not in generic and len(word) > 1), '')
    names = [brand] if brand else []
    # Preserve short multiword brand names without treating their generic tail
    # as standalone evidence (e.g. HS Fiber, Think Technology).
    if brand and brand in words:
        index = words.index(brand)
        if index + 1 < len(words):
            names.append(' '.join(words[index:index + 2]))
    return names


NETWORK_DEVICE = re.compile(
    r'\b(?:gpon|epon|xg[ -]?pon|xgs[ -]?pon|gepon|ont|onu|olt|router|routers|'
    r'roteador(?:es)?|gateway|optical network (?:terminal|unit)|optical line terminal|'
    r'terminacao de rede|unidade de rede optica|terminal de linha optica)\b', re.I)
NETWORK_INTERFACE = re.compile(
    r'\b(?:ethernet|wi[ -]?fi|wlan|lan|wan|ssid|802\.11|pppoe|dhcp|vlan)\b', re.I)
WRONG_DEVICE = re.compile(
    r'\b(?:camera|cameras|camcorder|smartphone|mobile phone|cell(?:ular)? phone|'
    r'telefone celular|furadeira|parafusadeira|drill|chainsaw|motosserra|'
    r'angle grinder|power tool|serra eletrica)\b', re.I)


def validate_document(text, maker, model, strict=False):
    """Require identity in PDF text, with tolerant spelling and context checks."""
    content = folded(text)
    matches = list(flexible_pattern(model).finditer(content))
    if not matches:
        raise ValueError('Model not found as a complete identifier in PDF text')
    brands = manufacturer_names(maker)
    brand = next((name for name in brands if flexible_pattern(name).search(content)), None)
    if strict and not brand:
        raise ValueError('Manufacturer/brand not found in PDF text')
    if strict and not MANUAL.search(text):
        raise ValueError('No manual/guide/datasheet wording in PDF text')
    # Prefer the title/cover and model context over unrelated devices mentioned
    # in later connection examples. Exclude distant copyright/footer text.
    windows = [content[max(0, match.start() - 120):match.end() + 180]
               for match in matches[:10]]
    if WRONG_DEVICE.search(content[:700]) and not NETWORK_DEVICE.search(content[:700]):
        raise ValueError('Document opening identifies a camera, phone or tool')
    def distance(left, right):
        return max(0, left.start() - right.end(), right.start() - left.end())

    # A camera title followed by instructions to connect it to a router must
    # not qualify just because the word "router" appears on that same page.
    first_model = matches[0]
    wrong_distance = min((distance(first_model, match) for match in WRONG_DEVICE.finditer(content)),
                         default=float('inf'))
    device_distance = min((distance(first_model, match) for match in NETWORK_DEVICE.finditer(content)),
                          default=float('inf'))
    if wrong_distance < 120 and wrong_distance < device_distance:
        raise ValueError('Model context identifies a camera, phone or tool rather than network equipment')
    if any(WRONG_DEVICE.search(window) for window in windows) and not any(
            NETWORK_DEVICE.search(window) for window in windows):
        raise ValueError('Model context identifies a different product category')
    interfaces = {match.group(0) for match in NETWORK_INTERFACE.finditer(content)}
    if not NETWORK_DEVICE.search(content) and len(interfaces) < 2:
        raise ValueError('Insufficient networking context in PDF text')
    return (f'Model identifier and networking context found; manufacturer: {brand or "unconfirmed"}; '
            f'document wording: {"manual/guide/datasheet" if MANUAL.search(text) else "technical document"}')


def search_queries(maker, model):
    """Try model-only queries before adding the manufacturer's first word."""
    words = maker.split()
    first_word = words[0] if words else ''
    model = ' '.join(model.split()).replace('"', '')
    brands = manufacturer_names(maker)
    first_word = brands[0] if brands else first_word
    identities = list(dict.fromkeys(['', first_word]))
    seen = set()
    for identity in identities:
        for suffix in ('', 'manual filetype:pdf', 'user guide'):
            prefix = identity if identity and not flexible_pattern(identity).search(folded(model)) else ''
            query = f'{prefix} {model} {suffix}'.strip()
            if query not in seen:
                seen.add(query)
                yield query


def device_key(maker, model):
    return (maker.strip().casefold(), model.strip().casefold())


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def checkpoint(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def pdf_text(data):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    if not reader.pages:
        raise ValueError('PDF has no pages')
    # Covers identity on title pages and manufacturer details near the end.
    indices = list(dict.fromkeys([*range(min(20, len(reader.pages))),
                                  *range(max(0, len(reader.pages) - 3), len(reader.pages))]))
    return '\n'.join(reader.pages[index].extract_text() or '' for index in indices)


class SharedState:
    """Run-scoped caches and one shared, rate-limited search client."""
    def __init__(self, delay=1, search_timeout=8, backends=DEFAULT_BACKENDS):
        self.lock = threading.Lock()
        self.search_lock = threading.Lock()
        self.cache = {}
        self.directory = tempfile.TemporaryDirectory(prefix='manuals-cache-')
        self.delay = delay
        self.search_timeout = search_timeout
        self.backends = backends
        self.next_search = 0
        self.client = None
        self.backend_cooldowns = {}
        self.backend_empty_counts = {}

    def once(self, key, operation):
        with self.lock:
            owner = key not in self.cache
            future = self.cache.setdefault(key, Future())
        if owner:
            try:
                future.set_result(operation())
            except Exception as exc:
                future.set_exception(exc)
        else:
            LOG.info('CACHE %s', key)
        return future.result()

    def search(self, operation):
        # Serialize client access; downloads continue concurrently.
        with self.search_lock:
            time.sleep(max(0, self.next_search - time.monotonic()))
            self.next_search = time.monotonic() + self.delay
            return operation()

    def close(self):
        self.directory.cleanup()


class Downloader:
    def __init__(self, output, delay=1, max_mb=40, provider="ddgs", api_key=None, shared=None, strict=False, max_depth=2, max_links=30):
        from bs4 import BeautifulSoup
        self.soup = BeautifulSoup
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.shared = shared or SharedState(delay)
        self.strict = strict
        self.max_depth = max_depth
        self.max_links = max_links
        self.successful_searches = 0
        self.provider = provider
        self.api_key = api_key
        self.search_errors = []
        self.delay = delay
        self.max_bytes = max_mb * 1024 * 1024
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0 (compatible; EquipmentManualDownloader/2.0)'

    def fetch(self, url):
        def download():
            data, final_url = self._fetch(url)
            path = Path(self.shared.directory.name) / hashlib.sha256(url.encode()).hexdigest()
            path.write_bytes(data)
            return path, final_url
        path, final_url = self.shared.once(('fetch', url), download)
        return path.read_bytes(), final_url

    def _fetch(self, url):
        if urlsplit(url).scheme not in ('http', 'https'):
            raise ValueError('Unsupported URL scheme')
        LOG.info('FETCH %s', url)
        with self.session.get(url, timeout=(5, 15), stream=True) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > self.max_bytes:
                    raise ValueError('Response exceeds size limit')
            return bytes(data), response.url

    def search_request(self, query, results):
        if self.provider == 'brave':
            with self.session.get(
                'https://api.search.brave.com/res/v1/web/search',
                params={'q': query, 'count': min(results, 20)},
                headers={'X-Subscription-Token': self.api_key, 'Accept': 'application/json'},
                timeout=(5, self.shared.search_timeout),
            ) as response:
                response.raise_for_status()
                return [{'href': item['url']} for item in
                        response.json().get('web', {}).get('results', []) if item.get('url')]
        from ddgs import DDGS
        if self.shared.client is None:
            self.shared.client = DDGS(timeout=self.shared.search_timeout)
        from ddgs.engines import ENGINES
        configured = [name.strip() for name in self.shared.backends.split(',') if name.strip()]
        invalid = set(configured) - set(ENGINES['text'])
        if invalid:
            raise ValueError('Unavailable DDGS engines: ' + ', '.join(sorted(invalid)))
        failures = []
        available = False
        for backend in self.shared.backends.split(','):
            backend = backend.strip()
            if not backend or self.shared.backend_cooldowns.get(backend, 0) > time.monotonic():
                continue
            available = True
            try:
                matches = list(self.shared.client.text(query, max_results=results, backend=backend))
            except Exception as exc:
                if str(exc).strip().casefold().rstrip('.') == 'no results found':
                    matches = []
                else:
                    self.shared.backend_cooldowns[backend] = time.monotonic() + 120
                    LOG.warning('ENGINE %s failed; cooling down for 120s: %s', backend, exc)
                    failures.append(f'{backend}: {exc}')
                    continue
            if matches:
                self.shared.backend_empty_counts[backend] = 0
                LOG.info('ENGINE %s: %d results', backend, len(matches))
                return matches
            count = self.shared.backend_empty_counts.get(backend, 0) + 1
            self.shared.backend_empty_counts[backend] = count
            LOG.info('ENGINE %s: no results (%d consecutive)', backend, count)
            if count >= 3:
                self.shared.backend_cooldowns[backend] = time.monotonic() + 60
                self.shared.backend_empty_counts[backend] = 0
                LOG.warning('ENGINE %s: repeated empty responses; cooling down for 60s', backend)
        if failures or not available:
            raise RuntimeError(' | '.join(failures) or 'All search engines are cooling down')
        return []

    def search(self, maker, model, results):
        seen = set()
        for query in search_queries(maker, model):
            LOG.info('SEARCH [%s] attempt 1/1; query=%s', self.provider, query)
            try:
                matches = self.shared.once(
                    ('search', self.provider, query, results),
                    lambda: self.shared.search(lambda: self.search_request(query, results)),
                )
            except Exception as exc:
                # Empty results and failed requests both advance to the next query.
                if str(exc).strip().casefold().rstrip('.') == 'no results found':
                    LOG.info('SEARCH returned 0 results; continuing fallbacks')
                    continue
                LOG.warning('SEARCH failed (%s): %s', type(exc).__name__, exc)
                self.search_errors.append(f'{query}: {exc}')
                continue
            self.successful_searches += 1
            LOG.info('SEARCH returned %s results', len(matches))
            matches = sorted(matches, key=lambda item: '.pdf' not in item.get('href', '').lower())
            for result in matches:
                url = result.get('href', '')
                if url and url not in seen:
                    seen.add(url)
                    LOG.info('RESULT %s', url)
                    yield url

    def save_pdf(self, data, url, maker, model):
        if not data.startswith(b'%PDF-'):
            raise ValueError('Response is not a PDF')
        text = self.shared.once(('text', hashlib.sha256(data).hexdigest()), lambda: pdf_text(data))
        evidence = validate_document(text, maker, model, strict=self.strict)
        digest = hashlib.sha256(data).hexdigest()
        stem = re.sub(r'[^\w.-]+', '_', f'{maker}_{model}', flags=re.UNICODE)[:120]
        identity = hashlib.sha256(repr(device_key(maker, model)).encode()).hexdigest()[:8]
        path = self.output / f'{stem}_{identity}_{digest[:12]}.pdf'
        temporary = path.with_suffix('.part')
        temporary.write_bytes(data)
        temporary.replace(path)
        LOG.info('SAVED %s', path)
        return dict(status='downloaded_candidate', source_url=url,
                    path=str(path.resolve()), sha256=digest,
                    detail=evidence + '; review hardware revision before use.',
                    validation_version=VALIDATION_VERSION)

    def find(self, maker, model, results):
        errors = []
        self.search_errors = []
        self.successful_searches = 0
        visited = set()
        try:
            for url in self.search(maker, model, results):
                if url in visited:
                    continue
                visited.add(url)
                try:
                    data, final_url = self.fetch(url)
                    if data.startswith(b'%PDF-'):
                        return self.save_pdf(data, final_url, maker, model)
                    queue = [(data, final_url, 0)]
                    followed = 0
                    while queue and followed < self.max_links:
                        body, page_url, depth = queue.pop(0)
                        if depth >= self.max_depth:
                            continue
                        page = self.soup(body, 'html.parser')
                        links = []
                        for tag in page.select('a[href], iframe[src], embed[src], object[data]'):
                            target = urljoin(page_url, tag.get('href') or tag.get('src') or tag.get('data'))
                            label = tag.get_text(' ', strip=True) + ' ' + unquote(target)
                            if urlsplit(target).scheme in ('http', 'https') and (
                                    '.pdf' in target.lower() or MANUAL.search(label)
                                    or re.search(r'download|support|document', label, re.I)):
                                links.append(target)
                        links.sort(key=lambda link: '.pdf' not in link.lower())
                        for target in dict.fromkeys(links):
                            if target in visited or followed >= self.max_links:
                                continue
                            visited.add(target)
                            followed += 1
                            try:
                                candidate, candidate_url = self.fetch(target)
                                if candidate.startswith(b'%PDF-'):
                                    return self.save_pdf(candidate, candidate_url, maker, model)
                                queue.append((candidate, candidate_url, depth + 1))
                            except Exception as exc:
                                LOG.info('REJECTED %s: %s', target, exc)
                                errors.append(f'{target}: {exc}')

                except Exception as exc:
                    LOG.info('REJECTED %s: %s', url, exc)
                    errors.append(f'{url}: {exc}')
        except Exception as exc:
            return dict(status='search_error', detail=str(exc)[:2000])
        if self.search_errors and not self.successful_searches:
            return dict(status='search_error', detail=' | '.join(self.search_errors + errors)[-4000:])
        return dict(status='not_found', detail=' | '.join(errors)[-4000:] or 'No matching manual found after all search queries')


def cached(row):
    path = Path(row.get('path') or '__missing__')
    if (row.get('validation_version') != VALIDATION_VERSION
            or row.get('status') != 'downloaded_candidate' or not path.is_file()):
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == row.get('sha256')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='input_sources/onts_onus_olts_roteadores_filtrados.csv')
    parser.add_argument('--output', default='input_sources/manuals')
    parser.add_argument('--report', default='input_sources/relatorio_download_manuais.csv')
    parser.add_argument('--limit', type=int, help='Maximum distinct equipment pairs to process')
    parser.add_argument('--results', type=int, default=6, help='Search results per query')
    parser.add_argument('--delay', type=float, default=1, help='Minimum seconds between search starts')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--search-timeout', type=float, default=8)
    parser.add_argument('--backends', default=DEFAULT_BACKENDS,
                        help='Ordered DDGS fallback engines; Yahoo first by default')
    parser.add_argument('--max-mb', type=int, default=40)
    parser.add_argument('--provider', choices=['ddgs', 'brave'], default='ddgs')
    parser.add_argument('--log-file', default='data/manuals_search.log')
    parser.add_argument('--strict-validation', action='store_true', help='Also require manufacturer and manual wording')
    parser.add_argument('--max-depth', type=int, default=2, help='Maximum linked-page depth')
    parser.add_argument('--max-links', type=int, default=30, help='Maximum links fetched per search result')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--revalidate', action='store_true',
                        help='Recheck successful local PDFs; search again if validation fails')
    args = parser.parse_args(argv)
    if args.max_depth < 1 or args.max_links < 1 or not args.backends.strip():
        parser.error("Depth, link budget, and backend list must be nonempty/positive")
    if args.workers < 1 or args.search_timeout <= 0 or args.delay < 0 or args.results < 1 or args.max_mb < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Limits must be positive and delay must be nonnegative')
    api_key = os.environ.get('BRAVE_API_KEY')
    if args.provider == 'brave' and not api_key:
        parser.error('--provider brave requires the BRAVE_API_KEY environment variable')
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s [%(threadName)s] %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding='utf-8')],
        force=True,
    )
    # Keep third-party HTTP/debug logging out of the query log.
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('primp').setLevel(logging.WARNING)
    logging.getLogger('ddgs').setLevel(logging.WARNING)
    LOG.info('Starting provider=%s; report=%s', args.provider, args.report)
    records = read_csv(args.input)
    if not records or not {'Nome do Fabricante', 'Modelo'}.issubset(records[0]):
        parser.error('Input must contain Nome do Fabricante and Modelo columns')
    previous = read_csv(args.report) if Path(args.report).exists() else []
    for row in previous:
        old_path = Path(row.get('path') or '__missing__')
        moved = Path(args.output) / old_path.name
        if not old_path.is_file() and moved.is_file():
            row['path'] = str(moved.resolve())
    report = {device_key(row['Nome do Fabricante'], row['Modelo']): row for row in previous}
    devices = dict.fromkeys((row['Nome do Fabricante'].strip(), row['Modelo'].strip()) for row in records)
    seen = set()
    pending = []
    for maker, model in devices:
        key = device_key(maker, model)
        if key in seen:
            continue
        seen.add(key)
        old = report.get(key)
        if old and ((not args.revalidate and not args.strict_validation and cached(old))
                    or (old.get('status') != 'downloaded_candidate' and not args.retry_failed)):
            continue
        if args.limit is not None and len(pending) >= args.limit:
            break
        pending.append((maker, model))
    prior_report = dict(report)
    shared = SharedState(args.delay, args.search_timeout, args.backends)
    local = threading.local()

    def process(maker, model):
        LOG.info('EQUIPMENT %s / %s', maker, model)
        if not maker or not model:
            return dict(status='missing_identity', detail='Manufacturer or model is empty')
        if not hasattr(local, 'downloader'):
            local.downloader = Downloader(args.output, args.delay, args.max_mb,
                                          args.provider, api_key, shared, args.strict_validation,
                                          args.max_depth, args.max_links)
        old = prior_report.get(device_key(maker, model), {})
        if old.get('status') == 'downloaded_candidate' and old.get('path'):
            path = Path(old['path'])
            LOG.info('REVALIDATION START %s / %s: %s', maker, model, path)
            try:
                data = path.read_bytes()
            except OSError as exc:
                LOG.warning('REVALIDATION UNREADABLE %s: %s; searching again', path, exc)
            else:
                try:
                    if not data.startswith(b'%PDF-'):
                        raise ValueError('Response is not a PDF')
                    text = shared.once(('text', hashlib.sha256(data).hexdigest()), lambda: pdf_text(data))
                    evidence = validate_document(text, maker, model, strict=args.strict_validation)
                except Exception as exc:
                    LOG.warning('REVALIDATION REJECTED %s / %s: %s; reason=%s', maker, model, path, exc)
                    LOG.info('REVALIDATION PRESERVED %s for review', path)
                else:
                    LOG.info('REVALIDATION PASSED %s / %s: %s; %s', maker, model, path, evidence)
                    # Reuse the verified local file; no copy or download needed.
                    return dict(status='downloaded_candidate', source_url=old.get('source_url', ''),
                                path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(),
                                detail='Revalidation passed: ' + evidence,
                                validation_version=VALIDATION_VERSION)
        return local.downloader.find(maker, model, args.results)

    processed = 0
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='manual')
    futures = {}
    try:
        futures = {executor.submit(process, maker, model): (maker, model)
                   for maker, model in pending}
        for future in as_completed(futures):
            maker, model = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                LOG.exception('Worker failed for %s / %s', maker, model)
                result = dict(status='worker_error', detail=str(exc))
            row = dict.fromkeys(FIELDS, '')
            row.update({'Nome do Fabricante': maker, 'Modelo': model}, **result)
            report[device_key(maker, model)] = row
            checkpoint(args.report, report.values())
            LOG.info('EQUIPMENT RESULT %s / %s: %s', maker, model, row['status'])
            processed += 1
            print(f'[{processed}/{len(pending)}] {maker} / {model}: {row["status"]}', flush=True)
    finally:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        shared.close()
    print(f'Report: {args.report}; processed {processed} equipment pairs.')


if __name__ == '__main__':
    main()

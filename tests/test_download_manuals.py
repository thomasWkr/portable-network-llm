import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from unittest.mock import patch

import download_manuals as dm


class DownloadTests(unittest.TestCase):
    def test_discovery_rejects_html_and_wrong_pdf_then_downloads(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = dm.Downloader(folder, delay=0)
            worker.search = lambda *args: iter(['https://example.com/manual.pdf', 'https://www.nokia.com/support'])
            page = b'<a href="/wrong.pdf">Manual</a><a href="/right.pdf">Manual</a>'
            responses = {
                'https://www.nokia.com/support': page,
                'https://www.nokia.com/wrong.pdf': b'%PDF-wrong',
                'https://www.nokia.com/right.pdf': b'%PDF-right',
            }
            worker.fetch = lambda url: (responses[url], url)
            with patch.object(dm, 'pdf_text', side_effect=['Other model manual', 'Maker AX100 Router User Manual']):
                row = worker.find('Maker', 'AX100', 5)
            self.assertEqual(row['status'], 'downloaded_candidate')
            self.assertTrue(dm.cached(row))
            Path(row['path']).write_bytes(b'changed')
            self.assertFalse(dm.cached(row))
            with self.assertRaises(ValueError):
                worker.save_pdf(b'<html>error</html>', 'https://www.nokia.com', 'Maker', 'AX100')

    def test_public_fetch_rejects_unapproved_redirect_before_requesting_it(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = dm.Downloader(folder, delay=0)
            self.addCleanup(worker.shared.close)
            with patch.object(dm.requests, 'Session') as session:
                public = session.return_value.__enter__.return_value
                response = public.get.return_value.__enter__.return_value
                response.is_redirect = True
                response.headers = {'Location': 'https://nokia.com.example.org/manual.pdf'}
                with self.assertRaisesRegex(ValueError, 'approved official HTTPS'):
                    worker.fetch('https://www.nokia.com/manual.pdf')
                public.get.assert_called_once_with('https://www.nokia.com/manual.pdf',
                                                   timeout=(5, 15), stream=True, allow_redirects=False)
                public.cookies.clear.assert_called_once()
                self.assertFalse(public.trust_env)

    def test_only_official_https_urls_without_credentials_are_allowed(self):
        self.assertTrue(dm.official_manual_url('https://static.tp-link.com/manual.pdf'))
        for url in ('http://www.nokia.com/manual.pdf', 'https://nokia.com.example.org/manual.pdf',
                    'https://user:pass@nokia.com/manual.pdf', 'https://nokia.com/manual.pdf?token=abc',
                    'https://fccid.io/manual.pdf'):
            with self.subTest(url=url):
                self.assertFalse(dm.official_manual_url(url))

    def test_empty_search_continues_to_next_query(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = dm.Downloader(folder, delay=0)
            with patch('ddgs.DDGS') as engine:
                engine.return_value.text.side_effect = [
                    *[Exception('No results found.')] * len(dm.DEFAULT_BACKENDS.split(',')),
                    [{'href': 'https://example.com/manual.pdf'}],
                ]
                search = worker.search('Maker Co., Ltd.', 'AX100', 6)
                self.assertEqual(next(search), 'https://example.com/manual.pdf')
                query = engine.return_value.text.call_args.args[0]
                self.assertEqual(query, 'AX100 manual filetype:pdf')
                self.assertEqual(worker.search_errors, [])

    def test_queries_include_short_manufacturer_and_model_only(self):
        queries = list(dm.search_queries('Maker Technologies Co., Ltd.', 'AX100'))
        self.assertEqual(queries, [
            'AX100', 'AX100 manual filetype:pdf', 'AX100 user guide',
            'maker AX100', 'maker AX100 manual filetype:pdf', 'maker AX100 user guide',
        ])
        self.assertEqual(list(dm.search_queries('', 'AX100')), queries[:3])

    def test_transient_search_failure_advances_without_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = dm.Downloader(folder, delay=0)
            with patch.object(worker, 'search_request') as search_request:
                search_request.side_effect = [
                    TimeoutError('timeout'), [{'href': 'https://example.com/manual.pdf'}],
                ]
                self.assertEqual(next(worker.search('Maker', 'AX100', 6)),
                                 'https://example.com/manual.pdf')
                queries = [call.args[0] for call in search_request.call_args_list]
                self.assertEqual(queries, ['AX100', 'AX100 manual filetype:pdf'])
                self.assertEqual(len(worker.search_errors), 1)

    def test_empty_results_and_transport_errors_have_different_statuses(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = dm.Downloader(folder, delay=0)
            with patch.object(dm, 'search_queries', side_effect=lambda maker, model: iter([model])), patch('ddgs.DDGS') as engine:
                engine.return_value.text.side_effect = Exception('No results found.')
                self.assertEqual(worker.find('Maker', 'AX100', 6)['status'], 'not_found')
                engine.return_value.text.side_effect = TimeoutError('timeout')
                self.assertEqual(worker.find('Maker', 'OTHER', 6)['status'], 'search_error')

    def test_brave_response_and_search_logging(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = dm.Downloader(folder, delay=0, provider='brave', api_key='test-secret')
            with patch.object(worker.session, 'get') as get:
                response = get.return_value.__enter__.return_value
                response.json.return_value = {'web': {'results': [{'url': 'https://example.com/manual.pdf'}]}}
                with self.assertLogs('manuals', level='INFO') as logs:
                    self.assertEqual(next(worker.search('Maker', 'AX100', 6)),
                                     'https://example.com/manual.pdf')
                self.assertIn('query=AX100', '\n'.join(logs.output))
                self.assertIn('SEARCH returned 1 results', '\n'.join(logs.output))
                self.assertNotIn('test-secret', '\n'.join(logs.output))
                self.assertEqual(get.call_args.kwargs['headers']['X-Subscription-Token'], 'test-secret')

    def test_concurrent_fetch_is_downloaded_once(self):
        shared = dm.SharedState(delay=0)
        self.addCleanup(shared.close)
        with tempfile.TemporaryDirectory() as folder:
            workers = [dm.Downloader(folder, delay=0, shared=shared) for _ in range(3)]
            def response(url):
                time.sleep(0.02)
                return b'%PDF-test', url
            with patch.object(dm.Downloader, '_fetch', side_effect=response) as fetch:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    values = list(executor.map(lambda worker: worker.fetch('https://example.com/a.pdf'), workers))
                self.assertEqual(fetch.call_count, 1)
                self.assertEqual(values, [(b'%PDF-test', 'https://example.com/a.pdf')] * 3)

    def test_search_client_reuse_pdf_priority_and_cache(self):
        shared = dm.SharedState(delay=0)
        self.addCleanup(shared.close)
        with tempfile.TemporaryDirectory() as folder, patch('ddgs.DDGS') as engine:
            engine.return_value.text.return_value = [
                {'href': 'https://example.com/support'}, {'href': 'https://example.com/a.pdf'}]
            workers = [dm.Downloader(folder, delay=0, shared=shared) for _ in range(2)]
            for worker in workers:
                self.assertEqual(next(worker.search('Maker', 'AX100', 6)), 'https://example.com/a.pdf')
            self.assertEqual(engine.call_count, 1)
            self.assertEqual(engine.return_value.text.call_count, 1)
            self.assertEqual(engine.return_value.text.call_args.kwargs['backend'], 'yahoo')

    def test_shared_search_throttle(self):
        shared = dm.SharedState(delay=0.02)
        self.addCleanup(shared.close)
        times = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(lambda _: shared.search(lambda: times.append(time.monotonic())), range(3)))
        self.assertTrue(all(b - a >= 0.015 for a, b in zip(times, times[1:])))

    def test_validation_accepts_brand_abbreviations_and_format_variations(self):
        examples = [
            ('Shenzhen C-Data Technology Co., Ltd.', 'FD511G-X',
             'C DATA FD 511G X GPON ONU User Guide'),
            ('Intelbras S.a. Industria de Telecomunicacao', 'AX100',
             'INTELBRAS AX-100 Manual do roteador. Conecte cameras a rede.'),
            ('TP-LINK Technologies Co., Ltd.', 'AX100',
             'TP Link AX100 Datasheet Ethernet LAN WAN'),
            ('Huawei Technologies Co. Ltd', 'HG8245',
             'HG8245 Optical Network Terminal User Manual Huawei'),
        ]
        for maker, model, text in examples:
            with self.subTest(maker=maker):
                self.assertIn('manufacturer', dm.validate_document(text, maker, model))

    def test_validation_rejects_wrong_identity_or_category(self):
        examples = [
            ('Other AX100 Router User Manual', 'Manufacturer'),
            ('Maker AX1000 Router User Manual', 'Model'),
            ('Maker AX100 Camera User Manual. Ethernet LAN WAN', 'camera'),
            ('Maker AX100 Mobile Phone User Guide. Wi-Fi WLAN', 'phone'),
            ('Maker AX100 Drill Instruction Manual', 'tool'),
            ('Maker AX100 Camera User Manual. Connect to your router with Ethernet.', 'camera'),
            ('Maker AX100 User Manual', 'networking'),
            ('Technology AX100 Router User Manual', 'Manufacturer'),
        ]
        for text, reason in examples:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, reason):
                dm.validate_document(text, 'Maker Technologies Co., Ltd.', 'AX100', strict=True)

    def test_old_downloads_are_not_trusted_without_new_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.pdf'
            path.write_bytes(b'%PDF-old')
            row = {'status': 'downloaded_candidate', 'path': str(path),
                   'source_url': 'https://www.nokia.com/manual.pdf',
                   'sha256': dm.hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertFalse(dm.cached(row))
            row['validation_version'] = dm.VALIDATION_VERSION
            self.assertTrue(dm.cached(row))

    def test_old_wrong_document_is_preserved_and_rejection_logged(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.csv'
            report = Path(folder) / 'report.csv'
            old = Path(folder) / 'old.pdf'
            old.write_bytes(b'%PDF-old')
            source.write_text('Nome do Fabricante,Modelo\nMaker,AX100\n')
            row = dict.fromkeys(dm.FIELDS, '')
            row.update({'Nome do Fabricante': 'Maker', 'Modelo': 'AX100',
                        'status': 'downloaded_candidate', 'path': str(old),
                        'source_url': 'https://www.nokia.com/manual.pdf',
                        'sha256': dm.hashlib.sha256(old.read_bytes()).hexdigest()})
            dm.checkpoint(report, [row])
            with patch.object(dm, 'pdf_text', return_value='Maker AX100 Camera User Manual'), \
                 patch.object(dm.Downloader, 'find', return_value={'status': 'not_found'}) as find:
                dm.main(['--input', str(source), '--report', str(report), '--output', folder,
                         '--log-file', str(Path(folder) / 'log')])
            find.assert_called_once()
            self.assertEqual(dm.read_csv(report)[0]['status'], 'not_found')
            self.assertTrue(old.exists())
            log = (Path(folder) / 'log').read_text()
            self.assertIn('REVALIDATION REJECTED', log)
            self.assertIn('REVALIDATION PRESERVED', log)

    def test_revalidate_forces_current_success_to_be_checked_locally(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.csv'
            report = Path(folder) / 'report.csv'
            pdf = Path(folder) / 'manual.pdf'
            pdf.write_bytes(b'%PDF-test')
            source.write_text('Nome do Fabricante,Modelo\nMaker,AX100\n')
            row = dict.fromkeys(dm.FIELDS, '')
            row.update({'Nome do Fabricante': 'Maker', 'Modelo': 'AX100',
                        'status': 'downloaded_candidate', 'path': str(pdf),
                        'source_url': 'https://www.nokia.com/manual.pdf',
                        'sha256': dm.hashlib.sha256(pdf.read_bytes()).hexdigest(),
                        'validation_version': dm.VALIDATION_VERSION})
            dm.checkpoint(report, [row])
            args = ['--input', str(source), '--report', str(report), '--output', folder,
                    '--log-file', str(Path(folder) / 'log')]
            with patch.object(dm, 'pdf_text', return_value='Maker AX100 Router User Manual') as text, \
                 patch.object(dm.Downloader, 'find') as find:
                dm.main(args)
                text.assert_not_called()
                dm.main(args + ['--revalidate'])
                text.assert_called_once()
                find.assert_not_called()
            self.assertEqual(dm.read_csv(report)[0]['status'], 'downloaded_candidate')
            self.assertTrue(pdf.exists())
            self.assertIn('REVALIDATION PASSED', (Path(folder) / 'log').read_text())

    def test_relaxed_validation_accepts_unbranded_technical_document(self):
        evidence = dm.validate_document('AX100 GPON terminal Ethernet specifications', 'Maker', 'AX100')
        self.assertIn('unconfirmed', evidence)
        with self.assertRaises(ValueError):
            dm.validate_document('AX100 GPON terminal Ethernet specifications', 'Maker', 'AX100', strict=True)
        with self.assertRaises(ValueError):
            dm.validate_document('AX1000 GPON terminal Ethernet specifications', 'Maker', 'AX100')

    def test_engine_failure_cools_down_and_uses_next_backend(self):
        shared = dm.SharedState(delay=0, backends='yahoo,brave')
        self.addCleanup(shared.close)
        with tempfile.TemporaryDirectory() as folder, patch('ddgs.DDGS') as engine:
            worker = dm.Downloader(folder, shared=shared)
            engine.return_value.text.side_effect = [TimeoutError('timeout'),
                [{'href': 'https://example.com/a.pdf'}], [{'href': 'https://example.com/b.pdf'}]]
            self.assertTrue(worker.search_request('AX100', 6))
            self.assertTrue(worker.search_request('AX200', 6))
            self.assertEqual([call.kwargs['backend'] for call in engine.return_value.text.call_args_list],
                             ['yahoo', 'brave', 'brave'])

    def test_unavailable_engine_does_not_trigger_automatic_search(self):
        with tempfile.TemporaryDirectory() as folder, patch('ddgs.DDGS') as engine:
            worker = dm.Downloader(folder, shared=dm.SharedState(backends='nonexistent'))
            self.addCleanup(worker.shared.close)
            with self.assertRaisesRegex(ValueError, 'Unavailable DDGS'):
                worker.search_request('AX100', 6)
            engine.return_value.text.assert_not_called()

    def test_repeated_empty_results_cool_down_engine(self):
        with tempfile.TemporaryDirectory() as folder, patch('ddgs.DDGS') as engine:
            worker = dm.Downloader(folder, shared=dm.SharedState(backends='brave'))
            self.addCleanup(worker.shared.close)
            engine.return_value.text.return_value = []
            for _ in range(3):
                self.assertEqual(worker.search_request('AX100', 6), [])
            with self.assertRaisesRegex(RuntimeError, 'cooling down'):
                worker.search_request('AX100', 6)
            self.assertEqual(engine.return_value.text.call_count, 3)

    def test_discovers_embedded_pdf_through_download_page(self):
        with tempfile.TemporaryDirectory() as folder:
            worker = dm.Downloader(folder, delay=0)
            worker.search = lambda *args: iter(['https://www.nokia.com/support'])
            responses = {'https://www.nokia.com/support': b'<a href="/download">Download</a>',
                         'https://www.nokia.com/download': b'<iframe src="/manual.pdf"></iframe>',
                         'https://www.nokia.com/manual.pdf': b'%PDF-test'}
            worker.fetch = lambda url: (responses[url], url)
            with patch.object(dm, 'pdf_text', return_value='AX100 GPON Ethernet specifications'):
                self.assertEqual(worker.find('Maker', 'AX100', 6)['status'], 'downloaded_candidate')

    def test_real_pdf_parser_rejects_corrupt_pdf(self):
        with self.assertRaises(Exception):
            dm.pdf_text(b'%PDF-corrupt')

    def test_device_workers_overlap_and_report_contains_all_results(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.csv'
            report = Path(folder) / 'report.csv'
            source.write_text('Nome do Fabricante,Modelo\nMaker,A\nMaker,B\nMaker,C\n')
            barrier = threading.Barrier(3)
            def find(*args):
                barrier.wait(timeout=3)
                return {'status': 'not_found'}
            with patch.object(dm.Downloader, 'find', side_effect=find):
                dm.main(['--input', str(source), '--report', str(report), '--output', folder,
                         '--log-file', str(Path(folder) / 'log'), '--workers', '3'])
            self.assertEqual({row['Modelo'] for row in dm.read_csv(report)}, {'A', 'B', 'C'})

    def test_checkpoint_and_resume_deduplicate_equipment(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.csv'
            report = Path(folder) / 'report.csv'
            source.write_text('Nome do Fabricante,Modelo\nMaker,AX100\nMaker,AX100\nOther,B\n')
            arguments = ['--input', str(source), '--report', str(report), '--output', folder,
                         '--log-file', str(Path(folder) / 'search.log')]
            with patch.object(dm.Downloader, 'find', return_value={'status': 'not_found'}) as find:
                dm.main(arguments)
                self.assertEqual(find.call_count, 2)
                dm.main(arguments)
                self.assertEqual(find.call_count, 2)
                dm.main(arguments + ['--retry-failed', '--limit', '1'])
                self.assertEqual(find.call_count, 3)
            self.assertEqual(len(dm.read_csv(report)), 2)


if __name__ == '__main__':
    unittest.main()

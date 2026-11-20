import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

import rag_pipeline as rag


class RagTests(unittest.TestCase):
    def test_chunk_windows_cover_text_and_obey_token_limit(self):
        text = '\n\n'.join(' '.join(f'word{i}' for i in range(start, start + 20))
                           for start in range(0, 120, 20))
        tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        chunks = list(rag.page_chunks(text, tokenizer, max_tokens=30, overlap=5))
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(len(tokenizer.encode(chunk).ids), 30)
        words = {word for chunk in chunks for word in chunk.split()}
        self.assertEqual(words, set(text.split()))

    def test_margin_cleanup_preserves_values_and_commands(self):
        pages = [f'Company confidential\n\nDescription {i}\n192.168.1.1\nshow interface gpon\n\n{i}'
                 for i in range(1, 7)]
        cleaned = rag.clean_pages(pages)
        for page in cleaned:
            self.assertNotIn('Company confidential', page)
            self.assertIn('192.168.1.1', page)
            self.assertIn('show interface gpon', page)

    def test_hybrid_search_respects_equipment_and_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            db = rag.create_database(Path(folder) / 'index.sqlite')
            for number, (maker, model, collection) in enumerate([
                ('Acme', 'AX100', 'equipment'), ('Other', 'AX100', 'equipment'),
                ('', '', 'network')], 1):
                docid = str(number)
                db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?)',
                           (docid, docid, f'{docid}.pdf', collection, 'Title', 1))
                db.execute('INSERT INTO equipment VALUES(?,?,?,?,?)',
                           (docid, maker, model, rag.normalized(maker), rag.normalized(model)))
                text = 'GPON optical network configuration'
                db.execute('INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)',
                           (number, f'chunk{number}', docid, 1, '', text, 'passage: ' + text, 5))
                db.execute('INSERT INTO chunks_fts(rowid,text) VALUES(?,?)', (number, text))
            vectors = np.asarray([[1, 0], [1, 0], [.8, .6]], dtype=np.float32)
            results = rag.search_index(db, vectors, np.asarray([1, 0]), 'GPON',
                                       model='AX-100', manufacturer='Acme')
            self.assertEqual([item['document_id'] for item in results], ['1'])
            results = rag.search_index(db, vectors, np.asarray([1, 0]), 'GPON',
                                       model='AX100', manufacturer='Acme', include_network=True)
            self.assertEqual({item['document_id'] for item in results}, {'1', '3'})
            self.assertTrue(all(item['page'] == 1 for item in results))
            db.close()

    def test_removed_duplicate_retains_model_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = root / 'catalog.csv'
            catalog.write_text('Nome do Fabricante,Modelo\nAcme,AX100\nAcme,AX200\n')
            retained = root / 'Acme_AX100_aaaaaaaaaaaa.pdf'
            duplicate = root / 'Acme_AX200_aaaaaaaaaaaa.pdf'
            mapping = root / 'duplicate_files.json'
            mapping.write_text(json.dumps([{'kept': str(retained), 'removed': str(duplicate)}]))
            docs = [{'path': str(retained), 'collection': 'equipment', 'equipment': []}]
            rag.add_catalog_associations(docs, catalog, mapping)
            self.assertEqual({item['model'] for item in docs[0]['equipment']}, {'AX100', 'AX200'})

    def test_keyword_search_focuses_on_issue_after_equipment_selection(self):
        expression = rag.keyword_expression('Como configurar uma VLAN na ONU HG6145F?',
                                            model='HG6145F')
        self.assertIn('"VLAN"', expression)
        self.assertNotIn('HG6145F', expression)
        self.assertNotIn('"uma"', expression)

    def test_duplicate_sources_are_removed_and_equipment_aliases_survive_rebuild(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            equipment, network = root / 'equipment', root / 'network'
            equipment.mkdir(); network.mkdir()
            retained = equipment / 'Acme_AX100.pdf'
            duplicate = equipment / 'Acme_AX200.pdf'
            retained.write_bytes(b'%PDF same')
            duplicate.write_bytes(b'%PDF same')
            (network / 'two.pdf').write_bytes(b'%PDF same')
            report = root / 'report.csv'
            report.write_text(f'Nome do Fabricante,Modelo,path\nAcme,AX300,{duplicate}\n')
            catalog = root / 'catalog.csv'
            catalog.write_text('Nome do Fabricante,Modelo\nAcme,AX100\nAcme,AX200\n')
            docs = rag.discover([('equipment', equipment), ('network', network)], report)
            mapping = root / 'duplicate_files.json'
            rag.add_catalog_associations(docs, catalog, mapping)
            self.assertEqual([doc['path'] for doc in docs], [str(retained)])
            self.assertEqual({item['model'] for item in docs[0]['equipment']}, {'AX100', 'AX200', 'AX300'})
            self.assertEqual(rag.remove_duplicate_files(docs, report, mapping), 2)
            self.assertFalse(duplicate.exists())
            self.assertFalse((network / 'two.pdf').exists())
            docs = rag.discover([('equipment', equipment), ('network', network)], report)
            rag.add_catalog_associations(docs, catalog, mapping)
            self.assertEqual({item['model'] for item in docs[0]['equipment']}, {'AX100', 'AX200', 'AX300'})
            self.assertEqual(rag.remove_duplicate_files(docs, report, mapping), 0)


if __name__ == '__main__':
    unittest.main()

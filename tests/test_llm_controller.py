import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import llm_controller as llm
from mobile_telemetry import Telemetry
from summarize_runs import summarize


def evidence(number, page=None, text=None):
    return {'document_id': 'doc', 'chunk_id': f'doc:p{page or number}:c{number}',
            'path': 'manual.pdf', 'page': page or number, 'collection': 'equipment',
            'text': text or f'SENTINEL{number}', 'equipment': []}


class ControllerTests(unittest.TestCase):
    def test_stream_parsing_and_generation_metrics(self):
        events = [
            {'choices': [{'index': 0, 'delta': {'content': 'Uma ONU '}}]},
            {'choices': [{'index': 0, 'delta': {'content': 'termina o enlace [1].'}, 'finish_reason': 'stop'}]},
            {'usage': {'completion_tokens': 10}, 'timings': {'cache_n': 0}},
        ]
        wire = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events) + 'data: [DONE]\n\n'
        client = llm.LlamaClient()
        with patch.object(client, 'request', return_value=contextlib.closing(io.BytesIO(wire.encode()))):
            result = client.generate([{'role': 'user', 'content': 'question'}], 'test')
        self.assertEqual(result['answer_raw'], 'Uma ONU termina o enlace [1].')
        self.assertEqual(result['usage']['completion_tokens'], 10)
        self.assertIsNotNone(result['first_content_token_ms'])
        self.assertEqual(result['server_timings']['cache_n'], 0)

    def test_broken_stream_is_not_reported_as_completed(self):
        client = llm.LlamaClient()
        wire = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\ndata: [DONE]\n\n'
        with patch.object(client, 'request', return_value=contextlib.closing(io.BytesIO(wire))):
            with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
                client.generate([], 'test')

    def test_citations_use_real_file_and_page_not_generated_names(self):
        text, references, validation = llm.render_answer('Resposta [1], e outra [99].', [evidence(1)])
        self.assertEqual(validation['status'], 'invalid_ids')
        self.assertFalse(validation['semantic_support_verified'])
        self.assertIn('[referência inválida]', text)
        self.assertIn('manual.pdf — página PDF 1', text)
        self.assertEqual(len(references), 1)

    def test_prompt_drops_whole_passages_to_fit_actual_tokenizer_budget(self):
        client = SimpleNamespace(prompt_tokens=lambda messages: 10 + 20 * messages[0]['content'].count('SENTINEL'))
        messages, selected, tokens = llm.fit_prompt(client, 'Question', [evidence(i) for i in range(1, 4)], 100, 20)
        self.assertEqual(len(selected), 1)
        self.assertIn('SENTINEL1', messages[0]['content'])
        self.assertNotIn('SENTINEL2', messages[0]['content'])
        self.assertLessEqual(tokens + 20 + 32, 100)
        with self.assertRaisesRegex(ValueError, 'No complete evidence'):
            llm.fit_prompt(client, 'Question', [evidence(1)], 70, 20)

    def test_equipment_is_inferred_from_question_without_selection(self):
        retriever = llm.Retriever.__new__(llm.Retriever)
        retriever.models = ['HG6145F', 'HG6145F3', 'F660', 'AX-1800']
        self.assertEqual(retriever.infer_model('VLAN na HG6145F3?'), 'HG6145F3')
        self.assertEqual(retriever.infer_model('Como usar AX 1800?'), 'AX-1800')
        self.assertIsNone(retriever.infer_model('Comparar HG6145F e F660'))
        self.assertIsNone(retriever.infer_model('O que é GPON?'))

    def test_same_page_chunks_share_one_reference(self):
        first = evidence(1, page=2, text='A useful technical description ' + 'overlap ' * 5)
        second = evidence(2, page=2, text='overlap ' * 5 + 'with the next command')
        grouped = llm.merge_page_evidence([first, second, evidence(3, page=3)], 3)
        self.assertEqual(len(grouped), 2)
        self.assertEqual(len(grouped[0]['chunk_ids']), 2)
        self.assertEqual(grouped[0]['text'].count('overlap '), 5)

    def test_health_check_timeout_is_not_used_for_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            server = llm.ServerProcess.__new__(llm.ServerProcess)
            server.port, server.command, server.info = 8080, ['fake'], {}
            server.binary = Path(folder) / 'fake-binary'
            server.log_path = Path(folder) / 'server.log'
            server.process, server.handle = None, None
            client = SimpleNamespace(json=lambda endpoint: {'status': 'ok'}, timeout=2)
            with patch('llm_controller.socket.socket') as socket, \
                 patch('llm_controller.subprocess.Popen') as process, \
                 patch('llm_controller.LlamaClient', return_value=client):
                socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                process.return_value.poll.return_value = None
                self.assertEqual(server.start().timeout, 600)
                server.close()

    def test_benchmark_never_passes_gold_and_resumes_by_question_content(self):
        with tempfile.TemporaryDirectory() as folder:
            questions, log = Path(folder) / 'q.jsonl', Path(folder) / 'runs.jsonl'
            questions.write_text(json.dumps({'id': 'q1', 'question': 'First question?', 'answer': 'SECRET_GOLD'}) + '\n')
            calls = []
            controller = SimpleNamespace(experiment_id='experiment', run_config={})
            def answer(question):
                calls.append(question)
                return {'status': 'ok', 'experiment_id': 'experiment', 'question': question}
            controller.answer = answer
            args = SimpleNamespace(questions=str(questions), log=str(log), warmup=0, repeats=1, seed=42, cooldown=0)
            llm.benchmark(controller, args)
            llm.benchmark(controller, args)
            self.assertEqual(calls, ['First question?'])
            row = json.loads(log.read_text().splitlines()[0])
            self.assertEqual(row['gold']['answer'], 'SECRET_GOLD')
            questions.write_text(json.dumps({'id': 'q1', 'question': 'Changed question?'}) + '\n')
            llm.benchmark(controller, args)
            self.assertEqual(calls[-1], 'Changed question?')

    def test_battery_energy_is_estimated_only_with_valid_discharging_samples(self):
        sampler = Telemetry()
        sampler.samples = [
            {'t': 0, 'battery': {'status': 'Discharging', 'current_now': '-1000000', 'voltage_now': '4000000'},
             'controller_rss_kib': 100, 'server_rss_kib': 200},
            {'t': 2, 'battery': {'status': 'Discharging', 'current_now': '-2000000', 'voltage_now': '4000000'},
             'controller_rss_kib': 110, 'server_rss_kib': 250},
        ]
        self.assertAlmostEqual(sampler.result()['energy_joules_estimate'], 12)
        self.assertEqual(sampler.result()['server_peak_rss_kib'], 250)
        sampler.samples[-1]['battery']['status'] = 'Charging'
        self.assertIsNone(sampler.result()['energy_joules_estimate'])

    def test_summary_ignores_warmups_and_does_not_fabricate_accuracy_or_energy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runs.jsonl'
            rows = []
            for repeat, value in enumerate([100, 200]):
                rows.append({'experiment_id': 'x', 'question_id': 'q1', 'question_sha256': 'same',
                             'repeat': repeat, 'status': 'ok', 'config': {'model_id': 'model'},
                             'metrics': {'generation_ms': value}, 'telemetry': {'energy_joules_estimate': None}})
            rows.append(dict(rows[0], warmup=True))
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            result = summarize([path])[0]
            self.assertEqual(result['successful_runs'], 2)
            self.assertEqual(result['unique_questions'], 1)
            self.assertEqual(result['generation_ms_median'], 150)
            self.assertIsNone(result['accuracy'])
            self.assertIsNone(result['energy_joules_estimate_median'])


if __name__ == '__main__':
    unittest.main()

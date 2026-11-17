import csv
import json
from pathlib import Path
import tempfile
import unittest

from summarize_runs import export_answers


class AnswerExportTests(unittest.TestCase):
    def test_only_measured_answers_sorted_and_roundtrip_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'runs.jsonl', Path(folder) / 'out.csv'
            def row(identifier, answer, repeat=0, **extra):
                return dict(status='ok', experiment_id='e', question_id=identifier,
                            question=f'Questão {identifier}, com vírgula?', answer=answer,
                            repeat=repeat, **extra)
            records = [row('q10', 'Resposta antiga'), row('q2', 'Óptica, "GPON".\nFonte [1].'),
                       row('q10', 'Resposta atualizada'), row('q2', 'Aquecimento', warmup=True),
                       row('q2', 'Segunda repetição', repeat=1),
                       dict(status='error', question='Erro?', error='timeout')]
            source.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records))
            self.assertEqual(export_answers([source], output), 3)
            with output.open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ['question', 'answer'])
                result = list(reader)
            self.assertEqual(result[0]['answer'], 'Óptica, "GPON".\nFonte [1].')
            self.assertEqual(result[1]['answer'], 'Segunda repetição')
            self.assertEqual(result[2]['answer'], 'Resposta atualizada')
            self.assertEqual(export_answers([source], output, repeat=0), 2)

    def test_export_never_overwrites_input_log(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'runs.jsonl'
            source.write_text('')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                export_answers([source], source)
            self.assertEqual(source.read_text(), '')


if __name__ == '__main__':
    unittest.main()

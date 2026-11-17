"""Summarize benchmark JSONL without treating repeated questions as new coverage."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import re
import sys


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def summarize(paths):
    groups = defaultdict(dict)
    for path in paths:
        for number, line in enumerate(Path(path).read_text().splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('warmup') or 'experiment_id' not in row:
                continue
            key = (row.get('question_id'), row.get('question_sha256'), row.get('repeat'))
            if row.get('question_id') is None:
                key = (str(path), number)
            # Keep the latest attempt for resumed question/repetition pairs.
            groups[row['experiment_id']][key] = row
    summary = []
    for experiment_id, records in groups.items():
        rows = list(records.values())
        good = [row for row in rows if row.get('status') == 'ok']
        config = rows[-1].get('config', {})
        entry = {'experiment_id': experiment_id, 'model': config.get('model_id'),
                 'run_label': config.get('run_label'), 'rag': config.get('rag'),
                 'execution_mode': config.get('device', {}).get('execution_mode'),
                 'successful_runs': len(good), 'failed_runs': len(rows) - len(good),
                 'unique_questions': len({row.get('question_id') or row.get('question') for row in rows}),
                 'citation_id_errors': sum(row.get('citation_validation', {}).get('status') == 'invalid_ids' for row in good),
                 'output_limit_hits': sum(row.get('finish_reason') == 'length' for row in good),
                 'accuracy': None}
        for field in ('retrieval_ms', 'generation_ms', 'end_to_end_ms', 'first_content_token_ms'):
            values = [row['metrics'][field] for row in good if row.get('metrics', {}).get(field) is not None]
            entry[field + '_median'] = statistics.median(values) if values else None
            entry[field + '_p95'] = percentile(values, .95)
        for name, section, field in [('energy_joules_estimate', 'telemetry', 'energy_joules_estimate'),
                                     ('server_peak_rss_kib', 'telemetry', 'server_peak_rss_kib'),
                                     ('controller_peak_rss_kib', 'telemetry', 'controller_peak_rss_kib'),
                                     ('decode_tokens_per_second', 'server_timings', 'predicted_per_second')]:
            values = [row[section][field] for row in good if row.get(section, {}).get(field) is not None]
            entry[name + '_median'] = statistics.median(values) if values else None
            entry[name + '_samples'] = len(values)
        summary.append(entry)
    return summary


def export_answers(paths, output, repeat=None):
    """Export measured successful answers, preserving separate repetitions."""
    selected, experiments = {}, {}
    for path in paths:
        with Path(path).open(encoding='utf-8-sig') as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f'Skipping malformed JSON: {path}:{number}', file=sys.stderr)
                    continue
                if row.get('warmup') or row.get('status') != 'ok':
                    continue
                if repeat is not None and row.get('repeat') != repeat:
                    continue
                question, answer = row.get('question'), row.get('answer')
                if not isinstance(question, str) or not isinstance(answer, str):
                    raise ValueError(f'Missing question/answer text: {path}:{number}')
                experiment = row.get('experiment_id', '')
                experiments.setdefault(experiment, len(experiments))
                if row.get('question_id') is not None:
                    key = (experiment, str(row['question_id']), question, row.get('repeat'))
                else:
                    key = (str(path), number)
                selected[key] = row
    def order(row):
        identifier = row.get('question_id')
        natural_id = tuple((0, int(part)) if part.isdigit() else (1, part.casefold())
                           for part in re.split(r'(\d+)', str(identifier)) if part)
        return experiments[row.get('experiment_id', '')], natural_id, row.get('repeat') or 0
    rows = sorted(selected.values(), key=order)
    output = Path(output)
    if any(output.resolve() == Path(path).resolve() for path in paths):
        raise ValueError('CSV output must not overwrite an input log')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['question', 'answer'])
        writer.writeheader()
        writer.writerows({'question': row['question'], 'answer': row['answer']} for row in rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--csv', help='Write a flat metrics table')
    parser.add_argument('--qa-csv', help='Export only question and answer columns; exclude warm-ups and errors')
    parser.add_argument('--qa-repeat', type=int, help='Export only this zero-based repetition with --qa-csv')
    args = parser.parse_args()
    if args.qa_repeat is not None and (args.qa_repeat < 0 or not args.qa_csv):
        parser.error('--qa-repeat requires --qa-csv and a nonnegative repetition index')
    if args.qa_csv:
        count = export_answers(args.files, args.qa_csv, args.qa_repeat)
        print(f'Exported {count} question/answer pairs to {args.qa_csv}', file=sys.stderr)
        if not args.csv:
            return
    rows = summarize(args.files)
    if args.csv and rows:
        with Path(args.csv).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

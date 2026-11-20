"""Run prepared LLM presets sequentially; each child owns only its own server."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from llm_controller import DEFAULT_QUESTION_FILES, append_record, read_questions, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', action='append', help='Question file; repeat for multiple sets (default: broad and specific)')
    parser.add_argument('--llms', default='all')
    parser.add_argument('--config', default='configs/llms.json')
    parser.add_argument('--run-label', required=True, help='For example s10-lite-proot or s20-proot')
    parser.add_argument('--output', default='data/runs/experiment')
    parser.add_argument('--model-dir', default='data/llm')
    parser.add_argument('--server-bin', default='.runtime/llama.cpp/build/bin/llama-server')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--cooldown', type=float, default=0, help='Seconds between measured questions')
    parser.add_argument('--model-cooldown', type=float, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--battery-file')
    parser.add_argument('--no-rag', action='store_true')
    parser.add_argument('--include-equipment-model', action='store_true',
                        help='Include modelo_alvo from each question file in the benchmark prompt')
    parser.add_argument('--filter-rag-by-equipment-model', action='store_true',
                        help='Use modelo_alvo from each question file to prioritize matching manuals in RAG')
    args = parser.parse_args()
    if args.no_rag and args.filter_rag_by_equipment_model:
        parser.error('--filter-rag-by-equipment-model requires RAG enabled')
    if min(args.repeats, args.context, args.max_tokens, args.threads, args.top_k) < 1 or min(args.warmup, args.cooldown, args.model_cooldown) < 0:
        parser.error('Invalid benchmark settings')
    presets = json.loads(Path(args.config).read_text())
    models = list(presets) if args.llms == 'all' else args.llms.split(',')
    if any(key not in presets for key in models):
        parser.error('Unknown model preset')
    question_files = [Path(path) for path in (args.questions or DEFAULT_QUESTION_FILES)]
    if len({path.stem for path in question_files}) != len(question_files):
        parser.error('Question files must have distinct names')
    for path in question_files:
        read_questions(path)
    random.Random(args.seed).shuffle(models)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    failures = 0
    jobs = [(model, questions) for model in models for questions in question_files]
    for index, (model, questions) in enumerate(jobs):
        batch_output = output / questions.stem if len(question_files) > 1 else output
        batch_output.mkdir(parents=True, exist_ok=True)
        log = batch_output / f'{model}.jsonl'
        model_path = Path(args.model_dir) / presets[model]['filename']
        info = {'timestamp': utc_now(), 'llm': model, 'run_label': args.run_label, 'question_set': questions.stem,
                'include_equipment_model': args.include_equipment_model,
                'filter_rag_by_equipment_model': args.filter_rag_by_equipment_model}
        if not model_path.is_file():
            info.update(status='missing_model', path=str(model_path))
            failures += 1
        else:
            command = [sys.executable, 'llm_controller.py', 'benchmark', '--llm', model,
                       '--questions', str(questions), '--config', args.config, '--model-dir', args.model_dir,
                       '--server-bin', args.server_bin, '--run-label', args.run_label, '--log', str(log),
                       '--repeats', str(args.repeats), '--warmup', str(args.warmup), '--seed', str(args.seed),
                       '--context', str(args.context), '--max-tokens', str(args.max_tokens),
                       '--threads', str(args.threads), '--top-k', str(args.top_k), '--cooldown', str(args.cooldown)]
            if args.battery_file:
                command += ['--battery-file', args.battery_file]
            if args.no_rag:
                command.append('--no-rag')
            if args.include_equipment_model:
                command.append('--include-equipment-model')
            if args.filter_rag_by_equipment_model:
                command.append('--filter-rag-by-equipment-model')
            append_record(output / 'model_status.jsonl', dict(info, status='started'))
            print(f'Running {model} / {questions.stem} ({index + 1}/{len(jobs)})', flush=True)
            with (batch_output / f'{model}.console.log').open('a') as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            info.update(status='completed' if result.returncode == 0 else 'runtime_error',
                        returncode=result.returncode, command=command)
            failures += result.returncode != 0
        append_record(output / 'model_status.jsonl', info)
        print(f'{model} / {questions.stem}: {info["status"]}', flush=True)
        if index + 1 < len(jobs) and args.model_cooldown:
            time.sleep(args.model_cooldown)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()

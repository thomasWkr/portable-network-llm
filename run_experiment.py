"""Run prepared LLM presets sequentially; each child owns only its own server."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from llm_controller import append_record, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', required=True)
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
    args = parser.parse_args()
    if min(args.repeats, args.context, args.max_tokens, args.threads, args.top_k) < 1 or min(args.warmup, args.cooldown, args.model_cooldown) < 0:
        parser.error('Invalid benchmark settings')
    presets = json.loads(Path(args.config).read_text())
    models = list(presets) if args.llms == 'all' else args.llms.split(',')
    if any(key not in presets for key in models):
        parser.error('Unknown model preset')
    from llm_controller import read_questions
    read_questions(args.questions)
    random.Random(args.seed).shuffle(models)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for index, model in enumerate(models):
        log = output / f'{model}.jsonl'
        model_path = Path(args.model_dir) / presets[model]['filename']
        info = {'timestamp': utc_now(), 'llm': model, 'run_label': args.run_label}
        if not model_path.is_file():
            info.update(status='missing_model', path=str(model_path))
            failures += 1
        else:
            command = [sys.executable, 'llm_controller.py', 'benchmark', '--llm', model,
                       '--questions', args.questions, '--config', args.config, '--model-dir', args.model_dir,
                       '--server-bin', args.server_bin, '--run-label', args.run_label, '--log', str(log),
                       '--repeats', str(args.repeats), '--warmup', str(args.warmup), '--seed', str(args.seed),
                       '--context', str(args.context), '--max-tokens', str(args.max_tokens),
                       '--threads', str(args.threads), '--top-k', str(args.top_k), '--cooldown', str(args.cooldown)]
            if args.battery_file:
                command += ['--battery-file', args.battery_file]
            if args.no_rag:
                command.append('--no-rag')
            append_record(output / 'model_status.jsonl', dict(info, status='started'))
            print(f'Running {model} ({index + 1}/{len(models)})', flush=True)
            with (output / f'{model}.console.log').open('a') as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            info.update(status='completed' if result.returncode == 0 else 'runtime_error',
                        returncode=result.returncode, command=command)
            failures += result.returncode != 0
        append_record(output / 'model_status.jsonl', info)
        print(f'{model}: {info["status"]}', flush=True)
        if index + 1 < len(models) and args.model_cooldown:
            time.sleep(args.model_cooldown)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()

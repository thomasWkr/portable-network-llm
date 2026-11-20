"""Package the offline controller for ARM64 phones or inspect an ADB device."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile


CODE = ['termux_battery_bridge.py', 'run_experiment.py', 'summarize_runs.py', 'prepare_llm_model.py',
        'examples/questions.jsonl', 'llm_controller.py', 'mobile_telemetry.py', 'rag_pipeline.py', 'requirements/requirements-rag.txt',
        'configs/llms.json', 'scripts/build_llama.sh', 'scripts/setup_phone_debian.sh',
        'docs/MOBILE.md', 'docs/RAG.md', 'docs/ENERGY.md']
RAG_FILES = ['index.sqlite', 'embeddings.npy', 'build_info.json',
             'model/embedding_model.json', 'model/tokenizer.json', 'model/tokenizer_config.json',
             'model/config.json', 'model/special_tokens_map.json', 'model/onnx/model.onnx']


def pack(args):
    from llm_controller import DEFAULT_QUESTION_FILES, read_questions
    from rag_pipeline import digest_file
    presets = json.loads(Path(args.config).read_text())
    selected = list(presets) if args.all_llms or args.llms == 'all' else args.llms.split(',')
    if not selected:
        raise ValueError('No model presets configured')
    unknown = [key for key in selected if key not in presets]
    if unknown:
        raise ValueError('Unknown model presets: ' + ', '.join(unknown))
    missing = [key for key in selected if not (Path(args.model_dir) / presets[key]['filename']).is_file()]
    if missing:
        raise ValueError('Missing weights for presets: ' + ', '.join(missing)
                         + '. Prepare them with prepare_llm_model.py --preset PRESET.')
    files = {name: Path(name) for name in CODE}
    for name in DEFAULT_QUESTION_FILES:
        read_questions(name)
        files[name] = Path(name)
    files['configs/llms.json'] = Path(args.config)
    for name in RAG_FILES:
        files['data/rag/' + name] = Path(args.index) / name
    for key in selected:
        name = presets[key]['filename']
        files['data/llm/' + name] = Path(args.model_dir) / name
    if args.include_pdfs:
        for folder in ('input_sources/manuals', 'input_sources/other_sources'):
            for path in Path(folder).rglob('*.pdf'):
                files[str(path)] = path
    for archive_name, path in files.items():
        if not path.is_file():
            raise ValueError(f'Missing file: {path}; prepare the model/runtime data first')
        if Path(archive_name).is_absolute() or '..' in Path(archive_name).parts:
            raise ValueError('Unsafe archive path')
    info = {'llms': selected, 'contains_native_binary': False, 'requires_setup_on_phone': True,
            'files': {name: {'sha256': digest_file(path), 'bytes': path.stat().st_size}
                      for name, path in files.items()}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.part')
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as archive:
        for name, path in files.items():
            archive.add(path, arcname='portable-network-llm/' + name, recursive=False)
    temporary.replace(output)
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f'Created {output}: {output.stat().st_size / 1024**2:.1f} MiB; models={selected}')
    print('The PC binary is excluded; build the pinned runtime on the phone with scripts/setup_phone_debian.sh.')


def inspect(args):
    command = ['adb'] + (['-s', args.serial] if args.serial else [])
    if not args.serial:
        print(subprocess.run(command + ['devices', '-l'], capture_output=True, text=True, check=True).stdout)
        return
    from mobile_telemetry import device_info
    info = device_info(args.serial)
    print(json.dumps(info, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    package = sub.add_parser('pack')
    models = package.add_mutually_exclusive_group()
    models.add_argument('--llms', default='qwen3.5-0.8b', help='Comma-separated prepared presets, or all')
    models.add_argument('--all-llms', action='store_true', help='Pack every preset in the model config')
    package.add_argument('--config', default='configs/llms.json')
    package.add_argument('--index', default='data/rag')
    package.add_argument('--model-dir', default='data/llm')
    package.add_argument('--include-pdfs', action='store_true')
    package.add_argument('--output', default='data/mobile/portable-network-llm.tar.gz')
    device = sub.add_parser('inspect')
    device.add_argument('--serial')
    args = parser.parse_args()
    (pack if args.command == 'pack' else inspect)(args)


if __name__ == '__main__':
    main()

"""Download a GGUF explicitly selected for the local controller."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, hf_hub_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', help='Preset key from configs/llms.json')
    parser.add_argument('--config', default='configs/llms.json')
    parser.add_argument('--repo', default='lmstudio-community/SmolLM2-360M-Instruct-GGUF')
    parser.add_argument('--file', default='SmolLM2-360M-Instruct-Q4_K_M.gguf')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--check-all', action='store_true', help='Verify all preset filenames without downloads')
    parser.add_argument('--check', action='store_true', help='List candidate Q4_K_M files without downloading')
    parser.add_argument('--output', default='data/llm')
    args = parser.parse_args()
    if args.check_all:
        presets = json.loads(Path(args.config).read_text())
        checked = []
        for key, item in presets.items():
            info = HfApi().model_info(item['repo'])
            files = [f.rfilename for f in info.siblings if 'q4_k_m' in f.rfilename.lower()]
            checked.append({'preset': key, 'revision': info.sha, 'configured_file_exists': item['filename'] in files, 'candidates': files})
        print(json.dumps(checked, indent=2))
        return
    if args.preset:
        selected = json.loads(Path(args.config).read_text())[args.preset]
        args.repo, args.file = selected['repo'], selected['filename']
        if args.revision == 'main':
            args.revision = selected.get('revision', 'main')
    metadata = HfApi().model_info(args.repo, revision=args.revision)
    revision = metadata.sha
    available = [item.rfilename for item in metadata.siblings if 'q4_k_m' in item.rfilename.lower()]
    if args.check:
        print(json.dumps({'repo': args.repo, 'revision': revision, 'files': available}, indent=2))
        return
    if args.file not in {item.rfilename for item in metadata.siblings}:
        raise ValueError(f'File {args.file} not in repository; Q4_K_M candidates: {available}')
    path = Path(hf_hub_download(repo_id=args.repo, filename=args.file, revision=revision,
                               local_dir=args.output, cache_dir='data/hf_cache'))
    with path.open('rb') as handle:
        if handle.read(4) != b'GGUF':
            raise ValueError('Downloaded artifact is not a GGUF file')
        handle.seek(0)
        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    info = {'repo': args.repo, 'revision': revision, 'filename': args.file,
            'path': str(path), 'sha256': sha, 'bytes': path.stat().st_size}
    path.with_suffix(path.suffix + '.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()

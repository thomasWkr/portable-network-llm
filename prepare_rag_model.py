"""Download a pinned, official multilingual E5 ONNX model for offline retrieval."""
import argparse
import json
from pathlib import Path

MODEL_ID = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/rag/model')
    args = parser.parse_args()
    output = Path(args.output)
    snapshot_download(
        repo_id=MODEL_ID, revision=REVISION, local_dir=output,
        cache_dir='data/hf_cache',
        allow_patterns=['onnx/model.onnx', 'tokenizer.json', 'tokenizer_config.json',
                        'config.json', 'special_tokens_map.json'],
    )
    (output / 'embedding_model.json').write_text(json.dumps({
        'model_id': MODEL_ID, 'revision': REVISION, 'onnx_file': 'onnx/model.onnx',
        'pooling': 'attention-mask mean', 'normalize': True, 'dimensions': 384,
        'max_length': 512, 'query_prefix': 'query: ', 'passage_prefix': 'passage: ',
    }, indent=2))
    print(f'Model prepared at {output.resolve()}')


if __name__ == '__main__':
    main()

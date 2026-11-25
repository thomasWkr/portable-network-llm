"""Run both benchmark question sets against the Gemini Developer API."""

import argparse
import json
import os
from pathlib import Path

from llm_controller import DEFAULT_QUESTION_FILES, append_record, make_messages, read_questions


def run_questions(client, model, max_tokens, batches, output):
    for name, questions in batches:
        log = output / name / f'{model}.jsonl'
        pending = {row['id']: row['question'] for row in questions}
        if log.exists():
            with log.open(encoding='utf-8') as handle:
                for line in handle:
                    record = json.loads(line)
                    identifier = record['question_id']
                    if pending.get(identifier) != record['question'] or not record['answer'].strip():
                        raise ValueError(f'{log}: invalid or changed question {identifier}')
                    del pending[identifier]
        print(f'{name}: {len(pending)} questions remaining', flush=True)
        for identifier, question in pending.items():
            prompt = make_messages(question, [], no_rag=True)[0]['content']
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={'max_output_tokens': max_tokens},
            )
            answer = (response.text or '').strip()
            if not answer:
                raise ValueError(f'{name}, question {identifier}: Gemini returned no answer text')
            append_record(log, {'question_id': identifier, 'question': question, 'answer': answer})
        print(f'{name}: {log}', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--model', default='gemini-3.1-flash-lite', help='Gemini API model ID')
    selection.add_argument('--list-models', action='store_true', help='List models supporting generateContent')
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--output', type=Path, default=Path('data/runs/gemini'))
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error('--max-tokens must be positive')
    if not args.model or Path(args.model).name != args.model or args.model in ('.', '..'):
        parser.error('--model must be a Gemini API model ID without path components')
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        parser.error('Set GEMINI_API_KEY before running')
    try:
        from google import genai
        from google.genai.errors import APIError
    except ImportError:
        parser.error('Install the Gemini SDK with python -m pip install google-genai')

    try:
        with genai.Client(api_key=api_key, vertexai=False) as client:
            if args.list_models:
                for model in client.models.list():
                    if model.name and 'generateContent' in (model.supported_actions or ()):
                        print(model.name.removeprefix('models/'))
                return
            batches = [(Path(path).stem, read_questions(path)) for path in DEFAULT_QUESTION_FILES]
            run_questions(client, args.model, args.max_tokens, batches, args.output)
    except APIError as error:
        hint = ''
        if error.code == 503 and not args.list_models:
            hint = ('\nRetry later to resume this model, or use --list-models '
                    'and rerun with --model MODEL_ID.\n')
        parser.exit(1, f'{error}\n{hint}')
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()

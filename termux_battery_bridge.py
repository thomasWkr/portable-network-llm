"""Run in native Termux, outside PRoot, to share Android battery readings."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def collect(command='termux-battery-status', timeout=10):
    try:
        completed = subprocess.run([command], capture_output=True, text=True, timeout=timeout, check=True)
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError('Battery command returned a non-object JSON value')
        return {'timestamp': time.time(), 'payload': payload, 'error': None}
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {'timestamp': time.time(), 'payload': {}, 'error': f'{type(exc).__name__}: {exc}'}


def save(path, reading):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(reading, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--timeout', type=float, default=10)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0:
        parser.error('Interval and timeout must be positive')
    count = 0
    try:
        while True:
            started = time.monotonic()
            reading = collect(timeout=args.timeout)
            save(args.output, reading)
            if args.once or count % 5 == 0 or reading['error']:
                print(json.dumps(reading, ensure_ascii=False), flush=True)
            if args.once:
                break
            count += 1
            time.sleep(max(0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print('Battery bridge stopped. The controller will reject stale readings.', flush=True)


if __name__ == '__main__':
    main()

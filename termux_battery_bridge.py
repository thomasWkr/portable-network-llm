"""Run in native Termux, outside PRoot, to share Android battery readings."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


BATTERY_COMMAND = 'termux-battery-status'
API_HELP = 'Check that the Termux:API Android app is installed from the same source as Termux.'


def collect(command=BATTERY_COMMAND, timeout=10):
    try:
        completed = subprocess.run([command], capture_output=True, text=True, timeout=timeout, check=True)
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError('Battery command returned a non-object JSON value')
        for field in ('current', 'voltage'):
            try:
                valid = math.isfinite(float(payload.get(field)))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f'Battery command did not return a numeric {field}')
        if not isinstance(payload.get('status'), str) or not payload['status'].strip():
            raise ValueError('Battery command did not return status')
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
    parser.add_argument('--setup', action='store_true',
                        help='Install the termux-api package with pkg if the battery command is missing')
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0:
        parser.error('Interval and timeout must be positive')
    count = 0
    try:
        if not shutil.which(BATTERY_COMMAND):
            if args.setup and shutil.which('pkg'):
                try:
                    subprocess.run(['pkg', 'install', '-y', 'termux-api'], check=True)
                except (OSError, subprocess.SubprocessError) as exc:
                    sys.exit(f'Could not install termux-api: {exc}')
            if not shutil.which(BATTERY_COMMAND):
                message = 'Run in native Termux, outside Debian. Install with pkg install termux-api or rerun with --setup.'
                save(args.output, {'timestamp': time.time(), 'payload': {}, 'error': message})
                sys.exit(message)
        while True:
            started = time.monotonic()
            reading = collect(timeout=args.timeout)
            save(args.output, reading)
            if args.once or count % 5 == 0 or reading['error']:
                print(json.dumps(reading, ensure_ascii=False), flush=True)
            if count == 0:
                if reading['error']:
                    sys.exit(f'Battery API startup check failed. {API_HELP}')
                if reading['payload']['status'].strip().upper() != 'DISCHARGING':
                    print('Unplug USB and the charger before measuring energy.', file=sys.stderr, flush=True)
            if args.once:
                break
            count += 1
            time.sleep(max(0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print('Battery bridge stopped. The controller will reject stale readings.', flush=True)


if __name__ == '__main__':
    main()

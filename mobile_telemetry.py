"""Best-effort whole-device battery estimates; unsupported metrics stay null."""
import os
import json
from pathlib import Path
import platform
import subprocess
import threading
import time


BATTERY_FIELDS = ('current_now', 'voltage_now', 'status', 'capacity', 'temp')


def read_rss_kib(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def android_device():
    return Path('/system/bin/getprop').exists()


def device_info(adb_serial=None):
    info = {'platform': platform.platform(), 'machine': platform.machine(), 'hostname': platform.node(),
            'execution_mode': 'phone-full-proot' if android_device() else 'local-desktop'}
    try:
        info['mem_total_kib'] = int(next(line for line in Path('/proc/meminfo').read_text().splitlines()
                                         if line.startswith('MemTotal:')).split()[1])
    except (OSError, ValueError, StopIteration):
        info['mem_total_kib'] = None
    if adb_serial:
        info['execution_mode'] = 'pc-retrieval-phone-llm'
        info['adb_serial'] = adb_serial
    for prop, key in [('ro.product.model', 'phone_model'), ('ro.board.platform', 'chipset'),
                      ('ro.build.version.release', 'android_version')]:
        command = ['adb', '-s', adb_serial, 'shell', 'getprop', prop] if adb_serial else ['/system/bin/getprop', prop]
        if adb_serial or android_device():
            try:
                info[key] = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                info[key] = None
    return info


class Telemetry:
    def __init__(self, server_pid=None, adb_serial=None, interval=.25, current_unit='uA', battery_file=None, battery_max_age=5):
        self.battery_file = battery_file
        self.battery_max_age = battery_max_age
        self.server_pid = server_pid
        self.adb_serial = adb_serial
        self.interval = interval
        self.current_unit = current_unit
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    def battery(self):
        if self.battery_file:
            return self.bridge_battery()
        if self.adb_serial:
            # Fixed command, no question/model text is interpolated into a shell.
            command = 'for f in current_now voltage_now status capacity temp; do cat /sys/class/power_supply/battery/$f 2>/dev/null || echo unavailable; done'
            try:
                result = subprocess.run(['adb', '-s', self.adb_serial, 'shell', command],
                                        capture_output=True, text=True, timeout=3, check=True)
                return dict(zip(BATTERY_FIELDS, result.stdout.splitlines()))
            except (OSError, subprocess.SubprocessError) as exc:
                return {'_source': 'adb-sysfs', '_error': f'ADB battery read failed: {exc}'}
        result, errors = {}, {}
        for field in BATTERY_FIELDS:
            try:
                result[field] = (Path('/sys/class/power_supply/battery') / field).read_text().strip()
            except OSError as exc:
                errors[field] = f'{type(exc).__name__}: {exc}'
        required = ['status', 'current_now', 'voltage_now']
        missing = [name for name in required if name not in result]
        if missing:
            result['_error'] = 'Battery sensors inaccessible: ' + '; '.join(f'{name}: {errors[name]}' for name in missing)
        result['_source'] = 'sysfs'
        return result

    def bridge_battery(self):
        try:
            reading = json.loads(Path(self.battery_file).read_text(encoding='utf-8'))
            timestamp = float(reading['timestamp'])
            age = time.time() - timestamp
            if not -1 <= age <= self.battery_max_age:
                raise ValueError(f'Stale battery bridge sample (age {age:.1f}s); keep the Termux collector running')
            if reading.get('error'):
                raise ValueError('Termux battery collector failed: ' + str(reading['error']))
            payload = reading['payload']
            if not isinstance(payload, dict):
                raise ValueError('Battery API payload is not a JSON object')
            missing = [name for name in ('current', 'voltage', 'status') if payload.get(name) is None]
            if missing:
                raise ValueError('Android battery API does not expose: ' + ', '.join(missing))
            # Termux:API BatteryStatus reports current in uA and voltage in mV.
            current, millivolts = float(payload['current']), float(payload['voltage'])
            return {'current_now': current, 'voltage_now': millivolts * 1000,
                    'status': str(payload.get('status', '')), 'plugged': payload.get('plugged'),
                    'capacity': payload.get('percentage', payload.get('level')),
                    '_source': 'termux-api-file', '_current_unit': 'uA',
                    '_timestamp': timestamp, '_age_seconds': age, '_raw_api': payload}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {'_source': 'termux-api-file', '_error': f'Battery bridge unavailable: {exc}'}

    def sample(self):
        self.samples.append({'t': time.monotonic(), 'battery': self.battery(),
                             'controller_rss_kib': read_rss_kib(os.getpid()),
                             'server_rss_kib': read_rss_kib(self.server_pid) if self.server_pid else None})

    def loop(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        self.sample()

    def result(self):
        powers, reason = [], None
        for sample in self.samples:
            battery = sample['battery']
            if battery.get('_error'):
                reason = battery['_error']
                break
            status = battery.get('status', '').casefold()
            if not status or status in ('unavailable', 'unknown'):
                reason = 'Battery status unavailable; charging state could not be determined'
                break
            if status != 'discharging':
                reason = f'Battery is not discharging (status={battery.get("status")})'
                break
            if battery.get('plugged') and battery['plugged'] != 'UNPLUGGED':
                reason = 'External power connected; battery current cannot estimate total device energy'
                break
            try:
                unit = battery.get('_current_unit', self.current_unit)
                amps = abs(float(battery['current_now'])) / (1e6 if unit == 'uA' else 1e3)
                volts = float(battery['voltage_now']) / 1e6
                if not 2.5 <= volts <= 5.5 or not 0 < amps <= 20:
                    raise ValueError('Unexpected sensor units/range')
                powers.append((sample['t'], amps * volts))
            except (KeyError, ValueError):
                reason = 'Current/voltage telemetry unavailable or units outside expected range'
                break
        distinct = {s['battery'].get('_timestamp') for s in self.samples if s['battery'].get('_timestamp') is not None}
        if self.battery_file and len(distinct) < 2 and not reason:
            reason = 'Insufficient distinct battery bridge samples during this request'
        if len(powers) < 2 and not reason:
            reason = 'Insufficient battery samples'
        energy = None if reason else sum((a[1] + b[1]) * .5 * (b[0] - a[0]) for a, b in zip(powers, powers[1:]))
        def peak(field):
            return max((s[field] for s in self.samples if s[field] is not None), default=None)
        return {'energy_joules_estimate': energy, 'energy_unavailable_reason': reason,
                'energy_scope': 'gross whole-device battery estimate; no idle subtraction',
                'battery_current_unit': 'uA' if self.battery_file else self.current_unit,
                'battery_backend': 'termux-api-file' if self.battery_file else 'adb-sysfs' if self.adb_serial else 'sysfs',
                'distinct_battery_samples': len(distinct) if self.battery_file else len(powers),
                'samples': len(self.samples),
                'controller_peak_rss_kib': peak('controller_rss_kib'),
                'server_peak_rss_kib': peak('server_rss_kib'), 'raw_samples': self.samples}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose battery access without loading an LLM.')
    parser.add_argument('--battery-file')
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--battery-current-unit', choices=['uA', 'mA'], default='uA')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    with Telemetry(battery_file=args.battery_file, current_unit=args.battery_current_unit) as telemetry:
        time.sleep(args.seconds)
    print(json.dumps(telemetry.result(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

import errno
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from mobile_telemetry import Telemetry
from termux_battery_bridge import collect, save


def sample(t, battery):
    return {'t': t, 'battery': battery, 'controller_rss_kib': 1, 'server_rss_kib': 2}


class BatteryBridgeTests(unittest.TestCase):
    def test_bridge_retries_transient_io_errors_and_rejects_persistent_ones(self):
        telemetry = Telemetry(battery_file='test')
        payload = {'current': -500000, 'voltage': 4000, 'status': 'DISCHARGING'}
        fresh = json.dumps({'timestamp': time.time(), 'payload': payload, 'error': None})
        error = OSError(errno.EIO, 'Input/output error')
        with patch('mobile_telemetry.time.sleep'), patch(
                'mobile_telemetry.Path.read_text', side_effect=[error, fresh]) as read:
            self.assertEqual(telemetry.battery()['current_now'], -500000)
            self.assertEqual(read.call_count, 2)
        with patch('mobile_telemetry.time.sleep'), patch(
                'mobile_telemetry.Path.read_text', side_effect=error) as read:
            self.assertIn('Input/output error', telemetry.battery()['_error'])
            self.assertEqual(read.call_count, 3)

    def test_empty_sensor_readings_are_not_reported_as_charging(self):
        telemetry = Telemetry()
        telemetry.samples = [sample(0, {}), sample(1, {})]
        result = telemetry.result()
        self.assertIsNone(result['energy_joules_estimate'])
        self.assertIn('status unavailable', result['energy_unavailable_reason'])
        with patch('mobile_telemetry.Path.read_text', side_effect=PermissionError('denied')):
            battery = telemetry.battery()
        self.assertIn('PermissionError', battery['_error'])
        self.assertIn('inaccessible', battery['_error'])

    def test_termux_units_are_converted_and_distinct_samples_integrate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'battery.json'
            payload = {'current': -500000, 'voltage': 4000, 'status': 'DISCHARGING', 'plugged': 'UNPLUGGED'}
            save(path, {'timestamp': time.time() - 1, 'payload': payload, 'error': None})
            telemetry = Telemetry(battery_file=path, current_unit='mA')
            first = telemetry.battery()
            save(path, {'timestamp': time.time(), 'payload': payload, 'error': None})
            second = telemetry.battery()
            self.assertEqual(first['voltage_now'], 4000000)
            telemetry.samples = [sample(0, first), sample(2, second)]
            result = telemetry.result()
            self.assertAlmostEqual(result['energy_joules_estimate'], 4)
            self.assertEqual(result['battery_current_unit'], 'uA')
            self.assertEqual(result['distinct_battery_samples'], 2)

    def test_stale_collector_and_missing_api_properties_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'battery.json'
            telemetry = Telemetry(battery_file=path)
            save(path, {'timestamp': time.time() - 100, 'payload': {}, 'error': None})
            self.assertIn('Stale', telemetry.battery()['_error'])
            save(path, {'timestamp': time.time(), 'payload': {'voltage': 4000, 'status': 'DISCHARGING'}, 'error': None})
            self.assertIn('does not expose: current', telemetry.battery()['_error'])
            save(path, {'timestamp': time.time(), 'payload': {}, 'error': 'API service unavailable'})
            self.assertIn('collector failed', telemetry.battery()['_error'])

    def test_repeated_poll_of_one_snapshot_is_not_two_measurements(self):
        telemetry = Telemetry(battery_file='test')
        battery = {'_timestamp': 123, '_current_unit': 'uA', 'current_now': -500000,
                   'voltage_now': 4000000, 'status': 'DISCHARGING', 'plugged': 'UNPLUGGED'}
        telemetry.samples = [sample(0, battery), sample(1, battery)]
        result = telemetry.result()
        self.assertIsNone(result['energy_joules_estimate'])
        self.assertIn('distinct', result['energy_unavailable_reason'])

    def test_plugged_phone_is_rejected_even_if_status_says_discharging(self):
        telemetry = Telemetry()
        battery = {'current_now': -500000, 'voltage_now': 4000000,
                   'status': 'DISCHARGING', 'plugged': 'PLUGGED_USB'}
        telemetry.samples = [sample(0, battery), sample(1, battery)]
        self.assertIn('External power', telemetry.result()['energy_unavailable_reason'])

    def test_collector_errors_are_explicit_instead_of_reusing_old_data(self):
        with patch('termux_battery_bridge.subprocess.run', side_effect=FileNotFoundError('missing helper')):
            reading = collect()
        self.assertEqual(reading['payload'], {})
        self.assertIn('FileNotFoundError', reading['error'])


if __name__ == '__main__':
    unittest.main()

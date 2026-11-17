# Battery access from the Android prototype

The imported S20 runs contain `battery: {}` for every sample. They do not establish whether the phone was charging: the old message combined unavailable sensors and charging into one reason. No joule estimate can be recovered from those historical logs.

PRoot's simulated root does not grant Android sensor permissions. The update provides precise diagnostics and an explicit Termux:API file bridge. It never infers energy from battery percentage or substitutes zero for an unsupported measurement.

## 1. Update the phone code

On the computer, with USB debugging authorized:

```bash
adb push data/mobile/energy-update.tar.gz /sdcard/Download/
```

If more than one device is attached, use `adb -s PHONE_SERIAL push ...`.

In the phone's native Termux terminal:

```bash
proot-distro login tcc --bind "$HOME/storage/downloads:/mnt/download"
cd /root/portable-network-llm
tar -xzf /mnt/download/energy-update.tar.gz --strip-components=1
cp termux_battery_bridge.py /mnt/download/
exit
```

This updates the controller/telemetry utilities and documentation. Models and the retrieval index are unchanged.

## 2. Verify Android's battery API outside Debian

Install the [Termux:API companion app](https://github.com/termux/termux-api#installation) from the same distribution/signing source as your Termux app. Installing only the command-line package is not sufficient when the companion service is absent. Do not uninstall your current Termux installation merely to switch sources; it contains the project and Debian environment.

In **native Termux**, not Debian:

```bash
pkg install python termux-api
termux-battery-status
```

The JSON must expose numeric `current` and `voltage`, plus `status`. Termux:API reports current in microamps and voltage in millivolts; the bridge converts voltage to the controller's microvolt representation. See the [official BatteryStatus implementation](https://github.com/termux/termux-api/blob/master/app/src/main/java/com/termux/api/apis/BatteryStatusAPI.java).

Disconnect USB and the charger before measuring. Expect `status: DISCHARGING` and `plugged: UNPLUGGED`. If the command hangs or omits current/voltage, the bridge cannot supply a valid joule estimate. Resolve the API installation/access or share its actual output for diagnosis; changing an LLM flag cannot manufacture unavailable sensor data.

## 3. Start the native collector

Keep this running in one native Termux session:

```bash
python "$HOME/storage/downloads/termux_battery_bridge.py" \
  --output "$HOME/storage/downloads/battery.json"
```

It writes readings atomically, about once per second, and prints occasional status snapshots. Failed API calls publish an explicit error; stopped collectors produce stale timestamps that the controller rejects.

## 4. Check access from another session

Open another Termux session and enter Debian:

```bash
proot-distro login tcc --bind "$HOME/storage/downloads:/mnt/download"
cd /root/portable-network-llm
.venv-rag/bin/python mobile_telemetry.py \
  --battery-file /mnt/download/battery.json --seconds 3
```

Check `energy_joules_estimate`, `energy_unavailable_reason`, `battery_backend`, and `distinct_battery_samples`. This standalone test measures its diagnostic interval, not LLM inference.

## 5. Run a new experiment using the bridge

In Debian, while the native collector remains running:

```bash
.venv-rag/bin/python llm_controller.py benchmark \
  --questions input_sources/questions.csv --llm qwen3.5-0.8b \
  --repeats 1 --warmup 1 --run-label s20-proot-energy \
  --battery-file /mnt/download/battery.json \
  --log data/runs/s20-qwen-energy.jsonl
```

The same `--battery-file` flag works for `ask`, `chat`, and `run_experiment.py`. Use `--no-rag` for the matching no-RAG condition and a separate result file.

The bridge uses a default maximum reading age of five seconds (`--battery-max-age` controls it in the controller). A request needs at least two distinct collector readings. Repeated polling of one snapshot is not treated as multiple independent measurements. Very short requests may therefore have no energy estimate.

This is still a coarse, gross whole-phone battery estimate during the request, excluding model initialization and without idle subtraction. Sensor refresh rate, API latency and collector overhead limit precision. Keep the collector running under the same conditions for all models and for an idle baseline. Phone validation is pending; unit tests verify conversions, integration, stale/error handling, and charging rejection, not your phone's sensor availability.

# Query-only LLM controller and phone prototype

The technician supplies only a question. The controller retrieves evidence, optionally recognizes an exact equipment identifier written in that question, packs complete passages to the loaded LLM's tokenizer/context limit, and generates a Portuguese answer. There is no equipment selector. General questions search both document collections.

The LLM emits numeric source IDs such as `[1]`; the controller resolves them to actual PDF names/pages. Invalid IDs are visibly marked and logged. This verifies reference IDs only, not whether every claim is supported. Raw answers remain in the experiment logs for evaluation. Prompts request paraphrasing, exact commands/values/units, clarification for ambiguous equipment, and abstention when evidence is insufficient. Small models can still repeat or make factual mistakes.

## Desktop check

The prepared environment, runtime, Qwen3.5-0.8B and SmolLM2 weights are already available in this workspace. For a fresh checkout:

```bash
python -m venv .venv-rag
.venv-rag/bin/pip install -r requirements/requirements-rag.txt
bash scripts/build_llama.sh
.venv-rag/bin/python prepare_llm_model.py --preset qwen3.5-0.8b
```

The runtime script builds a pinned llama.cpp commit with CPU inference. It does not install system dependencies; install CMake, Git and a C++ compiler first. The model helper pins repository revisions, records GGUF SHA-256/size, and uses the presets in `configs/llms.json`. Quantizations are community conversions of the chosen models; their provenance is recorded. All five presets use Q4_K_M. A filename check is not proof that a model fits or performs well on a phone.

```bash
.venv-rag/bin/python llm_controller.py list-models
.venv-rag/bin/python llm_controller.py ask 'Qual é a função da ONU em uma rede GPON?'
.venv-rag/bin/python llm_controller.py ask 'Quais recursos de VLAN a HG6145F suporta?'
.venv-rag/bin/python llm_controller.py chat
```

Qwen3.5-0.8B is the default generative model. To test the smallest baseline, add `--llm smollm2-360m`. Preset selection chooses the LLM for the experiment, not the technician's equipment. `chat` is a terminal loop of independent questions for now; it has no conversational memory. If asked to clarify a model, restate the question with the model in its text. `/sair` exits.

The controller owns and stops its local server. It refuses to replace an existing server on the same port; select another `--port` or explicitly use `--server-url`. The pinned build uses llama.cpp's `/apply-template`, `/tokenize` and streaming `/v1/chat/completions` endpoints. Generation is greedy, seed 42, repeat penalty 1.1, prompt caching disabled, and thinking requested off; the managed server also sets reasoning budget zero. CPU threads default to four, context to 4096 and output cap to 256 tokens. No vision/audio projector is loaded. These are a starting protocol, not the best settings for every model.

## Prepare the other selected models

Download only the presets you intend to test. Gemma E2B/E4B files are several GB; the E4B can exceed available RAM on these phones once the OS, retrieval encoder and context memory are included. Record load failures rather than silently changing quantization for one model.

```bash
.venv-rag/bin/python prepare_llm_model.py --preset gemma4-e2b
.venv-rag/bin/python prepare_llm_model.py --preset gemma4-e4b
.venv-rag/bin/python prepare_llm_model.py --preset qwen3.5-2b
.venv-rag/bin/python prepare_llm_model.py --preset smollm2-360m
```

`--check-all` checks filenames/revisions without downloading weights. The initial automated validation covers actual generation with Qwen3.5-0.8B and SmolLM2 on this computer. It does not establish phone compatibility or answer accuracy for all five models.

## Full offline execution on Android, without a custom app

The first prototype uses Termux + a Debian ARM64 PRoot environment so the existing Python/ONNX retriever and llama.cpp can both run on the phone. This is **not a native APK benchmark**: report the PRoot layer as part of the tested system. Do not mix its measurements with native inference and call them model-only results.

On the computer, create a package containing code, index, query encoder and the selected GGUF:

```bash
.venv-rag/bin/python mobile_tools.py pack
adb devices -l
adb -s PHONE_SERIAL push data/mobile/portable-network-llm.tar.gz /sdcard/Download/
```

The default package contains Qwen3.5-0.8B. Use `--llms qwen3.5-0.8b,smollm2-360m` or other prepared presets to include more. `--include-pdfs` includes original documents for later local viewing; without it, citations still display filenames/pages, and the evidence text is in the database. The package excludes the computer's x86 binary, caches, logs, and Python virtual environment. A checksum sidecar describes its files.

Install [Termux from its supported distribution](https://github.com/termux/termux-app#installation). In Termux, allow shared-storage access when Android asks:

```bash
termux-setup-storage
pkg install proot-distro
proot-distro install debian:bookworm --name tcc
proot-distro login tcc --bind "$HOME/storage/downloads:/mnt/download"
```

These installation commands target the current [PRoot-Distro OCI CLI](https://github.com/termux/proot-distro). Older versions offering distro aliases use `proot-distro install debian` and `proot-distro login debian` instead; check `proot-distro --help` before installing. The bind path makes the downloaded package available inside the container.

Inside Debian:

```bash
cd /root
tar -xzf /mnt/download/portable-network-llm.tar.gz
cd portable-network-llm
bash scripts/setup_phone_debian.sh
.venv-rag/bin/python llm_controller.py ask 'Qual é a função da ONU em uma rede GPON?' --run-label s10-lite-proot
```

Use `--run-label s20-proot` on the other phone. Initial setup needs internet for Debian packages, Python dependencies and pinned llama.cpp source. After setup, model, encoder and index are local: questions do not require internet or the computer. Do not rebuild the index on the phone; use the packaged frozen index. Compile the runtime on each phone rather than reusing binaries built for newer ARM instruction sets.

The server is bound to `127.0.0.1`; no public chat service is started. The Python CLI is the prototype interface. No custom Android application or root access is required for this route. Installation and real-device inference remain to be verified on the S10 Lite and S20.

## Reproducible question batches

Use JSONL, JSON arrays, or CSV with `id,question` (also accepts `pergunta`). Reference answers and other gold fields may be included: they are logged for later grading and never put in the LLM prompt. `examples/questions.jsonl` contains demonstration questions, not your final 50-question test set.

```bash
.venv-rag/bin/python llm_controller.py benchmark \
  --llm qwen3.5-0.8b --questions examples/questions.jsonl \
  --repeats 3 --warmup 1 --run-label s10-lite-proot \
  --log data/runs/s10-lite-qwen.jsonl
```

The encoder and LLM remain loaded for the batch. Warm-ups are marked and excluded from summaries; questions run independently in a seeded shuffled order. Run again with the same file/settings to resume completed question/repetition pairs. Changed question text, model artifact, index, prompt/controller version, hardware label or settings produces a new result identity. Errors are retried on resume. Full request/answer logs are fsynced after each question.

For prepared models, sequential orchestration is available:

```bash
.venv-rag/bin/python run_experiment.py \
  --questions examples/questions.jsonl --llms qwen3.5-0.8b,smollm2-360m \
  --run-label s10-lite-proot --repeats 3 --model-cooldown 30
```

Use your final test file and `--llms all` once all weights are present. Missing models and runtime/load failures are recorded; the runner does not download them or count them as successful comparisons. `--no-rag` supplies a separate baseline. Evidence packing can differ across tokenizers when a context limit is reached: compare logged evidence and use a common smaller `--top-k` or larger feasible context if identical evidence is required.

```bash
.venv-rag/bin/python summarize_runs.py data/runs/s10-lite-qwen.jsonl --csv data/runs/summary.csv
```

Every successful request records question, raw/display answer, evidence, source IDs, prompt, generation settings, model/runtime/index hashes, device metadata and metrics. `status=ok` means the request completed, not that the answer is correct. Accuracy stays empty until you grade the answers. Citation-ID validity is not semantic factuality. Output-cap hits are flagged; do not score a cut-off answer as a complete procedure.

## Metrics and limitations

- `retrieval_ms` includes query embedding and hybrid search. `query_embedding_ms` is also separate.
- `first_content_token_ms` measures from generation request to first content delta. `generation_ms` is the full streaming request; `end_to_end_ms` includes retrieval and prompt packing. The CLI buffers the final answer for citation rendering, so first-content timing is not time to visible terminal output.
- Model/server loading and controller initialization are logged separately. Runtime-reported prefill/decode throughput and actual token counts are retained; prompt cache use is observable (`cache_n`) even though it is requested off.
- Memory is sampled RSS for the controller and, when managed locally, the server process. It is a sampled peak, not a complete OS-wide memory measurement.
- Energy is a **gross whole-device battery estimate**, integrating current times voltage over the sampled interval. It is not model-only energy, not an external-meter measurement and not idle-subtracted. Raw samples and unavailable reasons are saved. Charging/unknown battery status or inaccessible sensors yields `null`, never a fabricated zero. USB-connected phones commonly charge, making this estimate unavailable.
- Linux sysfs current is assumed microamps and voltage microvolts. Some vendors expose different current units: verify raw readings and use `--battery-current-unit mA` only if appropriate. The software rejects implausible ranges rather than guessing units.

For comparable trials, use consistent brightness, charge range, connectivity, background activity and thermal conditions, and perform a separate idle measurement if reporting incremental energy. Keep the same runtime/backend and quantization category across models; report file size, RAM and failures. More repetitions improve timing/energy estimates but do not create more than 50 independent questions.

## Optional PC-assisted inference test

This mode measures a phone LLM while retrieval runs on the computer. It is useful for debugging, **not proof of a fully autonomous phone RAG**.

Start `llama-server` manually in the phone's Debian project:

```bash
.runtime/llama.cpp/build/bin/llama-server \
  -m data/llm/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf --alias qwen3.5-0.8b \
  --host 127.0.0.1 --port 8080 -c 4096 -t 4 -tb 4 -ngl 0 -np 1 \
  --jinja --reasoning-budget 0 --no-context-shift
```

On the computer:

```bash
adb -s PHONE_SERIAL forward tcp:8080 tcp:8080
.venv-rag/bin/python llm_controller.py ask 'Qual é a função da ONU?' \
  --server-url http://127.0.0.1:8080 --adb-serial PHONE_SERIAL --run-label s20-assisted
```

The logs explicitly label this `pc-retrieval-phone-llm`. Best-effort ADB sensor reads may be denied by the phone. Controller memory here is computer memory; remote server RSS is unavailable. Full phone tests should be run from the phone terminal as above.

## Export questions and answers only

Use the updated summary script to export exactly `question,answer`, retaining answer text and citation footers:

```bash
.venv-rag/bin/python summarize_runs.py data/runs/s20-qwen.jsonl \
  --qa-csv data/runs/s20-qwen-questions-answers.csv
```

Warm-ups and failed requests are excluded. Rows are sorted by question ID and repetition. Repeated measured runs are retained; add `--qa-repeat 0` for only the first measured repetition. Separate model/RAG experiments should use separate files because this export intentionally has only two columns. The CSV uses UTF-8 with BOM for accents in spreadsheet applications and quotes multiline answers correctly. Original logs remain unchanged.

If using an older mobile package, copy the updated `summarize_runs.py` to the phone project first; it does not require updating models or rebuilding the index.

## Missing energy readings in PRoot

If raw samples contain empty battery objects, use the [Termux:API energy bridge guide](ENERGY.md). The controller now supports `--battery-file` and reports missing sensors separately from charging. The small `data/mobile/energy-update.tar.gz` update avoids copying model files again.

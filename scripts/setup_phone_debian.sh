#!/usr/bin/env bash
# Run INSIDE the Debian PRoot container, after unpacking the mobile bundle.
set -euo pipefail
RAG_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$(id -u)" -ne 0 ] || [ ! -f /etc/debian_version ]; then
    echo 'Run this inside the Debian PRoot container, as its simulated root user.' >&2
    exit 1
fi
case "$(uname -m)" in
    aarch64|arm64) ;;
    *) echo 'This phone setup expects ARM64.' >&2; exit 1 ;;
esac
apt-get update
apt-get install -y python3 python3-venv python3-pip build-essential cmake git ca-certificates libgomp1
cd "$RAG_PROJECT_DIR"
python3 -m venv .venv-rag
.venv-rag/bin/pip install -r requirements/requirements-rag.txt
bash scripts/build_llama.sh
.venv-rag/bin/python llm_controller.py list-models

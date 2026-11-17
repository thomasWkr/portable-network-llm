#!/usr/bin/env bash
set -euo pipefail
RAG_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAG_LLAMA_REVISION=5676dfbe01e7ebe0387913680eee0ed39bf4a363
RAG_LLAMA_SOURCE="$RAG_PROJECT_DIR/.runtime/llama.cpp"
mkdir -p "$RAG_PROJECT_DIR/.runtime"
if [ ! -d "$RAG_LLAMA_SOURCE/.git" ]; then
    git clone --depth 1 --no-checkout https://github.com/ggml-org/llama.cpp.git "$RAG_LLAMA_SOURCE"
fi
if ! git -C "$RAG_LLAMA_SOURCE" diff --quiet; then
    echo 'Local llama.cpp modifications found; refusing to overwrite them.' >&2
    exit 1
fi
if ! git -C "$RAG_LLAMA_SOURCE" cat-file -e "$RAG_LLAMA_REVISION^{commit}" 2>/dev/null; then
    git -C "$RAG_LLAMA_SOURCE" fetch --depth 1 origin "$RAG_LLAMA_REVISION"
fi
git -C "$RAG_LLAMA_SOURCE" checkout --detach "$RAG_LLAMA_REVISION"
cmake -S "$RAG_LLAMA_SOURCE" -B "$RAG_LLAMA_SOURCE/build" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build "$RAG_LLAMA_SOURCE/build" --target llama-server -j "${RAG_BUILD_JOBS:-2}"
printf '%s\n' "$RAG_LLAMA_REVISION" > "$RAG_PROJECT_DIR/.runtime/llama-revision.txt"

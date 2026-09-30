#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
set -euo pipefail

usage() {
    cat <<'EOF'
Optional artifact setup ONLY. Never installs packages, builds, selects a GPU, or benchmarks.

Usage: bash prepare_ort_llama_benchmark.sh [--python EXECUTABLE] ACTION [options]
Actions:
  --help                 Print this help without invoking Python or downloading
  --plan                 Validate and print the pinned download plan; no network/writes
  --download             Deliberately download the pinned artifacts and write a manifest
Options:
  --python EXECUTABLE     Existing interpreter (default: python3 on PATH)
  --download-root PATH   Required destination for ort/ and gguf/ subdirectories
  --output-dir PATH      Required destination for artifact-manifest.json
  --ort-revision SHA     Full immutable commit (default: fc04c8f93df696602fd9f300a30d1bf2e3081347)
  --gguf-revision SHA    Full immutable commit (default: 78eb92a46fc37e6b524df991ed9aca9bc6aa7b80)

Exact selections: microsoft/Phi-4-mini-instruct-onnx gpu/gpu-int4-rtn-block-32/*
                  unsloth/Phi-4-mini-instruct-GGUF Phi-4-mini-instruct-Q4_K_M.gguf
Only --download imports huggingface_hub (historical version unknown).
Execute run_ort_llama_benchmark.py separately. See README.md for dependencies and commands.
EOF
}

if [[ $# -eq 0 ]]; then usage; exit 0; fi
for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then usage; exit 0; fi
done
PYTHON_BIN=python3
arguments=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || {
                echo "--python requires an executable" >&2; exit 2;
            }
            PYTHON_BIN="$2"; shift 2 ;;
        *) arguments+=("$1"); shift ;;
    esac
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PYTHON_BIN" "$SCRIPT_DIR/prepare_benchmark_artifacts.py" "${arguments[@]}"

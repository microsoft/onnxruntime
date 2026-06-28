#!/bin/bash
# =============================================================================
# run_gpt_oss.sh
# End-to-end pipeline for gpt-oss-20b QMoE on the local CUDA 13.0 stack:
#   (1) build ONNX Runtime              (delegates to .env/cuda_130.sh --build)
#   (2) build + install onnxruntime-genai against the freshly built ORT
#   (3) build the gpt-oss-20b INT4 QMoE ONNX model (genai model builder)
#   (4) run a verification generation   (examples/python/model-qa.py)
#
# Usage:
#   ./run_gpt_oss.sh [STEPS] [OPTIONS]
#
# Steps (if none given, ALL steps run in order):
#   --build-ort        Build ONNX Runtime and install its wheel into $VENV
#   --build-genai      Assemble ORT_HOME, build onnxruntime-genai, install wheel
#   --build-model      Build the gpt-oss-20b INT4 QMoE ONNX model
#   --verify           Run a verification generation against the built model
#   --all              Run every step (default)
#
# Options:
#   --block-size N     QMoE block size (default: 32)
#   --output DIR       Model output directory (default: derived from block size)
#   --prompt TEXT      Verification user prompt (default: capital-of-France probe)
#   -h | --help        Show this help and exit
#
# Most paths are overridable via environment variables (see CONFIG below).
# =============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
# CONFIG (override via environment if your layout differs)
# ----------------------------------------------------------------------------
ORT_DIR="${ORT_DIR:-/home/tianlei/onnxruntime}"
GENAI_DIR="${GENAI_DIR:-/home/tianlei/onnxruntime-genai}"
VENV="${VENV:-/home/tianlei/venv}"

CUDA_HOME="${CUDA_HOME:-/home/tianlei/cuda13.0}"
CUDNN_HOME="${CUDNN_HOME:-/home/tianlei/cudnn_9.19_cuda13}"
CUDAARCHS="${CUDAARCHS:-90}"            # H200 = sm_90

ORT_HOME="${ORT_HOME:-/home/tianlei/ort_home_cu130}"   # assembled include/+lib/ for genai
ORT_BUILD_DIR="${ORT_BUILD_DIR:-build/cu130}"          # relative to ORT_DIR
GENAI_BUILD_DIR="${GENAI_BUILD_DIR:-build/cu130}"      # relative to GENAI_DIR
BUILD_TYPE="${BUILD_TYPE:-Release}"

HF_CACHE="${HF_CACHE:-/home/tianlei/hf_cache}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
INT4_ALGO="${INT4_ALGO:-k_quant_mixed}"

BLOCK_SIZE="${BLOCK_SIZE:-32}"
USE_FP4=false                           # --fp4 toggles MXFP4 QMoE export + build
OUTPUT_DIR=""                           # resolved after arg parsing
PROMPT="${PROMPT:-What is the capital of France?}"

# ----------------------------------------------------------------------------
# Arg parsing
# ----------------------------------------------------------------------------
DO_ORT=false; DO_GENAI=false; DO_MODEL=false; DO_VERIFY=false; ANY_STEP=false

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; }

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --build-ort)   DO_ORT=true;    ANY_STEP=true ;;
        --build-genai) DO_GENAI=true;  ANY_STEP=true ;;
        --build-model) DO_MODEL=true;  ANY_STEP=true ;;
        --verify)      DO_VERIFY=true; ANY_STEP=true ;;
        --all)         DO_ORT=true; DO_GENAI=true; DO_MODEL=true; DO_VERIFY=true; ANY_STEP=true ;;
        --block-size)  BLOCK_SIZE="$2"; shift ;;
        --fp4)         USE_FP4=true ;;
        --output)      OUTPUT_DIR="$2"; shift ;;
        --prompt)      PROMPT="$2"; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
    shift
done

# Default: run everything
if [ "$ANY_STEP" = false ]; then
    DO_ORT=true; DO_GENAI=true; DO_MODEL=true; DO_VERIFY=true
fi

# MXFP4 mandates a block size of 32; force it so output naming is consistent.
if [ "$USE_FP4" = true ]; then
    BLOCK_SIZE=32
fi

# Resolve output dir if not explicitly set
if [ -z "$OUTPUT_DIR" ]; then
    if [ "$USE_FP4" = true ]; then
        OUTPUT_DIR="/home/tianlei/gptoss20b_fp4_qmoe"
    else
        OUTPUT_DIR="/home/tianlei/gptoss20b_int4_qmoe_bs${BLOCK_SIZE}_ropefix"
    fi
fi

ORT_WHEEL_GLOB="$ORT_DIR/$ORT_BUILD_DIR/$BUILD_TYPE/dist/onnxruntime_gpu-*.whl"
GENAI_WHEEL_GLOB="$GENAI_DIR/$GENAI_BUILD_DIR/$BUILD_TYPE/wheel/dist/onnxruntime_genai_cuda-*.whl"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
log()  { echo -e "\n==== $* ===="; }
fail() { echo -e "\n!!!! $* !!!!" >&2; exit 1; }

setup_runtime_env() {
    export CUDA_HOME CUDA_PATH="$CUDA_HOME" CUDNN_HOME CUDAARCHS
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$ORT_HOME/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"
}

activate_venv() {
    # shellcheck disable=SC1091
    source "$VENV/bin/activate" || fail "Could not activate venv at $VENV"
}

# ----------------------------------------------------------------------------
# Step 1: Build ONNX Runtime
# ----------------------------------------------------------------------------
build_ort() {
    log "🚀 [1/4] Building ONNX Runtime ($ORT_BUILD_DIR/$BUILD_TYPE)"
    [ -x "$ORT_DIR/.env/cuda_130.sh" ] || fail "Missing $ORT_DIR/.env/cuda_130.sh"
    # With --fp4, build the CUDA QMoE FP4 (MXFP4) kernels (CUDA Toolkit 12.8+).
    local fp4_env="OFF"
    [ "$USE_FP4" = true ] && fp4_env="ON"
    ( cd "$ORT_DIR" && USE_FP4_QMOE="$fp4_env" ./.env/cuda_130.sh --build ) || fail "ONNX Runtime build failed"

    local wheel
    wheel=$(ls $ORT_WHEEL_GLOB 2>/dev/null | head -1) || true
    [ -n "${wheel:-}" ] || fail "ORT wheel not found at $ORT_WHEEL_GLOB"

    log "📦 Installing ORT wheel into $VENV"
    activate_venv
    pip install "$wheel" --force-reinstall
    log "✅ [1/4] ONNX Runtime built and installed: $(basename "$wheel")"
}

# ----------------------------------------------------------------------------
# Step 2: Build onnxruntime-genai (assemble ORT_HOME first), then install
# ----------------------------------------------------------------------------
assemble_ort_home() {
    log "🧩 Assembling ORT_HOME at $ORT_HOME"
    rm -rf "$ORT_HOME"
    mkdir -p "$ORT_HOME/include" "$ORT_HOME/lib"

    # Public session headers (flat layout expected by genai) + CPU provider factory
    cp "$ORT_DIR"/include/onnxruntime/core/session/*.h   "$ORT_HOME/include/"
    cp "$ORT_DIR"/include/onnxruntime/core/session/*.inc "$ORT_HOME/include/" 2>/dev/null || true
    cp "$ORT_DIR"/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h "$ORT_HOME/include/"

    # Shared libraries from the ORT build (preserve symlinks for the SONAME chain)
    local b="$ORT_DIR/$ORT_BUILD_DIR/$BUILD_TYPE"
    cp -P "$b"/libonnxruntime.so*                  "$ORT_HOME/lib/"
    cp    "$b"/libonnxruntime_providers_cuda.so    "$ORT_HOME/lib/"
    cp    "$b"/libonnxruntime_providers_shared.so  "$ORT_HOME/lib/"
    [ -f "$ORT_HOME/lib/libonnxruntime.so" ] || fail "ORT_HOME assembly incomplete (no libonnxruntime.so)"
}

build_genai() {
    log "🚀 [2/4] Building onnxruntime-genai"
    [ -f "$ORT_DIR/$ORT_BUILD_DIR/$BUILD_TYPE/libonnxruntime.so" ] \
        || fail "ONNX Runtime not built yet. Run with --build-ort first."
    assemble_ort_home

    activate_venv
    setup_runtime_env
    ( cd "$GENAI_DIR" && python build.py \
        --use_cuda \
        --cuda_home "$CUDA_HOME" \
        --ort_home "$ORT_HOME" \
        --config "$BUILD_TYPE" \
        --build_dir "$GENAI_BUILD_DIR" \
        --parallel \
        --skip_tests \
        --skip_examples ) || fail "onnxruntime-genai build failed"

    local wheel
    wheel=$(ls $GENAI_WHEEL_GLOB 2>/dev/null | head -1) || true
    [ -n "${wheel:-}" ] || fail "genai wheel not found at $GENAI_WHEEL_GLOB"

    log "📦 Installing genai wheel into $VENV"
    pip install "$wheel" --force-reinstall --no-deps
    log "✅ [2/4] onnxruntime-genai built and installed: $(basename "$wheel")"
}

# ----------------------------------------------------------------------------
# Step 3: Build the gpt-oss-20b INT4 QMoE model
# ----------------------------------------------------------------------------
build_model() {
    log "🚀 [3/4] Building model: $MODEL_NAME -> $OUTPUT_DIR (QMoE block_size=$BLOCK_SIZE, fp4=$USE_FP4)"
    activate_venv
    setup_runtime_env
    # Run the builder from genai source so the fixed gptoss.py builder is used.
    export PYTHONPATH="$GENAI_DIR/src/python/py:${PYTHONPATH:-}"
    mkdir -p "$OUTPUT_DIR"
    # MXFP4 export adds use_fp4_moe=true (block size is forced to 32 by the builder).
    local extra_moe=""
    [ "$USE_FP4" = true ] && extra_moe="use_fp4_moe=true"
    ( cd "$GENAI_DIR" && python -m models.builder \
        -m "$MODEL_NAME" \
        -o "$OUTPUT_DIR" \
        -p int4 \
        -e cuda \
        -c "$HF_CACHE" \
        --extra_options \
            qmoe_block_size="$BLOCK_SIZE" \
            int4_algo_config="$INT4_ALGO" \
            $extra_moe ) || fail "Model build failed"
    [ -f "$OUTPUT_DIR/model.onnx" ] || fail "Model build produced no model.onnx in $OUTPUT_DIR"
    log "✅ [3/4] Model built: $OUTPUT_DIR"
}

# ----------------------------------------------------------------------------
# Step 4: Verify the model with a short generation
# ----------------------------------------------------------------------------
verify_model() {
    log "🚀 [4/4] Verifying model at $OUTPUT_DIR"
    [ -f "$OUTPUT_DIR/genai_config.json" ] || fail "Missing genai_config.json in $OUTPUT_DIR"
    activate_venv
    setup_runtime_env
    ( cd "$GENAI_DIR" && python examples/python/model-qa.py \
        -m "$OUTPUT_DIR" \
        -e cuda \
        --non_interactive \
        -up "$PROMPT" ) || fail "Verification run failed"
    log "✅ [4/4] Verification complete"
}

# ----------------------------------------------------------------------------
# Orchestrate
# ----------------------------------------------------------------------------
log "gpt-oss pipeline | block_size=$BLOCK_SIZE | output=$OUTPUT_DIR"
echo "  steps: build-ort=$DO_ORT build-genai=$DO_GENAI build-model=$DO_MODEL verify=$DO_VERIFY"

[ "$DO_ORT"    = true ] && build_ort
[ "$DO_GENAI"  = true ] && build_genai
[ "$DO_MODEL"  = true ] && build_model
[ "$DO_VERIFY" = true ] && verify_model

log "🎉 Done."

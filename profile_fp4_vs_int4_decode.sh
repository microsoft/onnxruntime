#!/bin/bash
# Profile FP4 vs INT4 QMoE gpt-oss-20b decode kernels with nsys.
# CUDA graph is forced OFF so individual kernels (dequant + GEMM) are attributable.
set -uo pipefail

GENAI_DIR="${GENAI_DIR:-/home/tianlei/onnxruntime-genai}"
ORT_VENV="${ORT_VENV:-/home/tianlei/venv}"
ORT_HOME="${ORT_HOME:-/home/tianlei/ort_home_cu130}"
CUDA_HOME="${CUDA_HOME:-/home/tianlei/cuda13.0}"
CUDNN_HOME="${CUDNN_HOME:-/home/tianlei/cudnn_9.19_cuda13}"
HF_HOME="${HF_HOME:-/home/tianlei/hf_cache}"
NSYS="${NSYS:-$CUDA_HOME/bin/nsys}"

FP4_MODEL="${FP4_MODEL:-/home/tianlei/gptoss20b_fp4_qmoe}"
INT4_MODEL="${INT4_MODEL:-/home/tianlei/gptoss20b_int4_qmoe_bs32_ropefix}"

PROMPT_LEN="${PROMPT_LEN:-64}"
GEN_LEN="${GEN_LEN:-64}"
OUT_DIR="${OUT_DIR:-/tmp/qmoe_fp4_profile}"
mkdir -p "$OUT_DIR"
export TMPDIR="$OUT_DIR/nsystmp"
mkdir -p "$TMPDIR"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME
export LD_LIBRARY_PATH="$ORT_HOME/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib64:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$ORT_VENV/bin/activate"

run_one() {
    local tag="$1" model="$2"
    local cfg="$model/genai_config.json"
    echo "=== profiling $tag : $model ==="
    cp "$cfg" "$cfg.profbak"
    sed -i 's/"enable_cuda_graph": "1"/"enable_cuda_graph": "0"/' "$cfg"

    local rep="$OUT_DIR/${tag}_decode"
    rm -f "$rep.nsys-rep" "$rep.sqlite"
    ( cd "$GENAI_DIR" && "$NSYS" profile \
        --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --force-overwrite=true -o "$rep" \
        python benchmark/python/benchmark_e2e.py \
            -i "$model" -e cuda \
            -b 1 -l "$PROMPT_LEN" -g "$GEN_LEN" -r 1 -w 1 \
            --use_random_tokens --chat_template "{input}" \
            -mn gpt-oss-20b -pr "$tag" -o "$OUT_DIR/${tag}.csv" ) \
        > "$OUT_DIR/${tag}_run.log" 2>&1

    mv "$cfg.profbak" "$cfg"
    grep -E "Throughput|Latency" "$OUT_DIR/${tag}_run.log" || true

    echo "--- top GPU kernels ($tag) ---"
    "$NSYS" stats --report cuda_gpu_kern_sum --format table "$rep.nsys-rep" 2>/dev/null \
        | head -30
}

run_one fp4  "$FP4_MODEL"
run_one int4 "$INT4_MODEL"

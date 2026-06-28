#!/bin/bash
# Benchmark/profile the FP4 QMoE gpt-oss-20b decode path with the fused MXFP4 GEMV
# (W4A16) decode path ENABLED vs DISABLED (the Phase 0 dequant fallback), on the
# SAME FP4 model. ORT_ENABLE_FP4_GEMV gates the new path; everything else is held
# fixed so the delta is attributable to the GEMV routing.
#
#   OFF: ORT_ENABLE_FP4_GEMV=0      -> top-k dequant fallback (QMoEDequantizeFp4WeightsKernel); GEMV is now the default
#   ON : ORT_ENABLE_FP4_GEMV=1      -> fused prologue/expand/GEMV/finalize, no per-fwd dequant
set -uo pipefail

GENAI_DIR="${GENAI_DIR:-/home/tianlei/onnxruntime-genai}"
ORT_VENV="${ORT_VENV:-/home/tianlei/venv}"
ORT_HOME="${ORT_HOME:-/home/tianlei/ort_home_cu130_fp4_bench}"
ORT_BUILD_ABS="${ORT_BUILD_ABS:-/home/tianlei/onnxruntime/build/cu130_fp4_bench/Release}"
CUDA_HOME="${CUDA_HOME:-/home/tianlei/cuda13.0}"
CUDNN_HOME="${CUDNN_HOME:-/home/tianlei/cudnn_9.19_cuda13}"
HF_HOME="${HF_HOME:-/home/tianlei/hf_cache}"
NSYS="${NSYS:-$CUDA_HOME/bin/nsys}"

FP4_MODEL="${FP4_MODEL:-/tianlei/models/gpt-oss-20b/cuda_int4_rtn_mixed_fp4_qmoe}"

PROMPT_LEN="${PROMPT_LEN:-64}"
GEN_LEN="${GEN_LEN:-128}"
REPS="${REPS:-3}"
WARMUP="${WARMUP:-2}"
OUT_DIR="${OUT_DIR:-/tmp/qmoe_fp4_gemv_onoff}"
mkdir -p "$OUT_DIR"
export TMPDIR="$OUT_DIR/nsystmp"
mkdir -p "$TMPDIR"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export HF_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$ORT_HOME/lib:$ORT_BUILD_ABS:$CUDA_HOME/lib64:$CUDNN_HOME/lib64:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source "$ORT_VENV/bin/activate"

# Force CUDA graph OFF so individual decode kernels are attributable in nsys.
cfg="$FP4_MODEL/genai_config.json"
cp "$cfg" "$cfg.profbak"
sed -i 's/"enable_cuda_graph": "1"/"enable_cuda_graph": "0"/' "$cfg"
restore_cfg() { mv -f "$cfg.profbak" "$cfg" 2>/dev/null || true; }
trap restore_cfg EXIT

run_one() {
    local tag="$1"; shift   # remaining args are env assignments (e.g. ORT_ENABLE_FP4_GEMV=1)
    echo "=== profiling $tag (env: $*) ==="
    local rep="$OUT_DIR/${tag}_decode"
    rm -f "$rep.nsys-rep" "$rep.sqlite"
    ( cd "$GENAI_DIR" && env "$@" "$NSYS" profile \
        --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --force-overwrite=true -o "$rep" \
        python benchmark/python/benchmark_e2e.py \
            -i "$FP4_MODEL" -e cuda \
            -b 1 -l "$PROMPT_LEN" -g "$GEN_LEN" -r "$REPS" -w "$WARMUP" \
            --use_random_tokens --chat_template "{input}" \
            -mn gpt-oss-20b -pr "$tag" -o "$OUT_DIR/${tag}.csv" ) \
        > "$OUT_DIR/${tag}_run.log" 2>&1

    echo "--- throughput/latency ($tag) ---"
    grep -E "Throughput|Latency|tps|tokens" "$OUT_DIR/${tag}_run.log" | head -20 || true
    if [[ -f "$OUT_DIR/${tag}.csv" ]]; then
        echo "--- csv ($tag) ---"; cat "$OUT_DIR/${tag}.csv"
    fi
    echo "--- top GPU kernels ($tag) ---"
    "$NSYS" stats --report cuda_gpu_kern_sum --format table "$rep.nsys-rep" 2>/dev/null \
        | head -25
}

run_one fp4_gemv_off ORT_ENABLE_FP4_GEMV=0
run_one fp4_gemv_on  ORT_ENABLE_FP4_GEMV=1

restore_cfg
echo "=== DONE. reports in $OUT_DIR ==="

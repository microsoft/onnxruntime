# gpt-oss-20b Inference Benchmark — ONNX Runtime GenAI vs llama.cpp vs vLLM

Single-GPU benchmark of `openai/gpt-oss-20b` comparing ONNX Runtime GenAI
(INT4 and FP4/MXFP4 QMoE) against llama.cpp (MXFP4 GGUF) and vLLM (native MXFP4).

## Hardware & software

| Item | Value |
| --- | --- |
| GPU | 1× NVIDIA H200 (sm_90 / Hopper, 143 771 MiB) |
| CUDA Toolkit | 13.0 |
| cuDNN | 9.19 (cuda13) |
| ONNX Runtime | 1.28.0 (CUDA EP, built from source) |
| onnxruntime-genai | 0.14.0-dev (`tlwu/model_builder_qwen3.6`) |
| llama.cpp | tag b9630 (CUDA, `-DCMAKE_CUDA_ARCHITECTURES=90`) |
| vLLM | 0.23.0 (torch 2.11.0+cu130) |

All runs pinned to one GPU (`CUDA_VISIBLE_DEVICES=0`), batch size 1.

## Model configurations

| Engine | Quantization | Notes |
| --- | --- | --- |
| ONNX Runtime GenAI | INT4 QMoE, block size 32 | `k_quant_mixed`, CUTLASS fpA_intB mixed GEMM |
| ONNX Runtime GenAI | FP4 (MXFP4) QMoE, block size 32 | `quant_type=fp4`, built with `onnxruntime_USE_FP4_QMOE=ON` |
| llama.cpp | MXFP4 GGUF | `gpt-oss-20b-mxfp4.gguf` (11.27 GiB) |
| vLLM | MXFP4 (native) | `openai/gpt-oss-20b` |

## Methodology

- **ONNX Runtime GenAI**: `benchmark/python/benchmark_e2e.py` with
  `--use_random_tokens --chat_template "{input}"` (pure throughput, no chat
  template formatting). Decode (token generation) and prefill (prompt
  processing) throughput are reported per token. CUDA graph toggled via
  `enable_cuda_graph` in `genai_config.json` (restored after each run).
- **llama.cpp**: `llama-bench -ngl 99` (all layers on GPU). CUDA graphs are on
  by default.
- **vLLM**: `vllm bench latency --batch-size 1`. vLLM reports end-to-end
  latency; decode throughput is estimated as `gen_len / avg_latency`, so the
  denominator includes prefill and slightly under-reports steady-state decode.
  Must be run from a neutral working directory so the installed package is not
  shadowed by the vLLM source tree.

## Results — prompt 512, generate 128 (batch 1)

| Engine | Prefill (tps) | Decode (tps) | Decode (ms/tok) | Notes |
| --- | ---: | ---: | ---: | --- |
| ORT GenAI — INT4 QMoE bs32 (graph OFF) | 21 343.0 | 244.3 | 4.09 | |
| ORT GenAI — INT4 QMoE bs32 (graph ON)  | 19 719.0 | 271.5 | 3.68 | best ORT decode |
| ORT GenAI — FP4 QMoE bs32 (graph OFF)  | 2 335.9 | 4.86 | 205.93 | MXFP4 dequant fallback (see below) |
| ORT GenAI — FP4 QMoE bs32 (graph ON)   | 2 313.7 | 4.80 | 208.54 | MXFP4 dequant fallback (see below) |
| llama.cpp — MXFP4                       | 9 577.0 | 297.1 | 3.37 | graphs on by default |
| vLLM — MXFP4 native                     | — | ~278.5\* | ~3.59\* | \*e2e latency incl. prefill |

## Results — prompt 128, generate 256 (batch 1)

| Engine | Prefill (tps) | Decode (tps) | Decode (ms/tok) | Notes |
| --- | ---: | ---: | ---: | --- |
| ORT GenAI — INT4 QMoE bs32 (graph OFF) | 9 555.0 | 251.8 | 3.97 | wall ≈ 374 tps |
| ORT GenAI — INT4 QMoE bs32 (graph ON)  | 9 002.0 | 273.4 | 3.66 | wall ≈ 405 tps |
| llama.cpp — MXFP4 (tg256)              | — | 298.5 | 3.35 | |

## Key findings

- **INT4 QMoE is the fastest ORT path on H200.** With CUDA graph enabled, ORT
  GenAI reaches ~271 decode tps, within ~9 % of llama.cpp (297 tps) and on par
  with vLLM's ~278 e2e tps. ORT's prefill throughput (~20k tps) is roughly 2×
  llama.cpp's at this shape.
- **CUDA graph helps INT4 decode (~+11 %)** but slightly reduces prefill on the
  variable-length prompt phase. llama.cpp and vLLM use graphs by default.
- **FP4 (MXFP4) QMoE is functionally correct but slow on Hopper.** The native
  MXFP4 tensor-core GEMM path requires Blackwell (sm_120); on H200 (sm_90) the
  QMoE op falls back to dequantizing MXFP4 weights to BF16 on every forward pass
  (`use_fp4_dequant_fallback_ = sm_ < 120` in
  [onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc](../onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc)).
  This makes decode ~50× slower than INT4 on this GPU. **FP4 QMoE is expected to
  be advantageous only on sm_120+ hardware.** The exported FP4 model generates
  correct output (verified: "The capital of France is **Paris**.").
  *(The 4.86 tps figure above is the original all-expert dequant fallback. An
  active-experts-only dequant optimization has since raised FP4 decode to ~163
  tps — see the gap analysis below.)*

## Gap analysis — why FP4 trails INT4 and llama.cpp on Hopper

After the active-experts-only dequant optimization (only the top-k experts a
token routes to are dequantized, instead of all 32), FP4 decode rose from 4.86
to ~163 tps. It still trails INT4 (~270 tps) and llama.cpp (~297 tps). Measured
on H200 (prompt 512, gen 128, CUDA graph ON):

| Engine | Prefill (tps) | Decode (tps) | Decode (ms/tok) |
| --- | ---: | ---: | ---: |
| ORT INT4 QMoE (fused W4A16) | 19 912 | 270.1 | 3.70 |
| ORT FP4 QMoE (dequant fallback) | 3 266 | 163.1 | 6.13 |
| llama.cpp MXFP4 (fused W4A8) | 9 577 | 297.1 | 3.37 |

The non-MoE layers (attention, projections, LM head) are byte-identical between
the INT4 and FP4 models, so the **entire gap is in the QMoE op**. nsys decode
profiles (gen129 − gen1, 128 isolated decode steps) attribute it cleanly:

| Decode MoE kernel | INT4 | FP4 fallback |
| --- | ---: | ---: |
| Fused `fpA_intB MoeFCGemm` (4-bit weights, in-register dequant) | dominant | — |
| Dense FP16 grouped GEMM on **dequantized** weights (`GemmUniversal`/`MoeFCGemm` FP16) | — | dominant (~52% of decode) |
| `QMoEDequantizeFp4` (MXFP4 → FP16 to HBM) | — | present, m-independent |
| `moe_gemv` (+ swiglu), finalize, expand, expert maps | small | small |

### Root cause: no fused MXFP4 kernel on sm_90 → a dequant round-trip through HBM

- **INT4** uses a true **fused weight-only mixed GEMM** (`fpA_intB`): the 4-bit
  weights stay 4-bit in HBM and are dequantized **in registers** inside the GEMM.
  One kernel, weights read once at 0.5 byte/element.
- **FP4** has **no fused MXFP4 GEMM** below sm_120, so the fallback **materializes
  the active experts' weights as FP16 in HBM** (`QMoEDequantizeFp4`) and then runs
  a **dense FP16 grouped GEMM** over them. Two kernels, and the weight matrix is
  written and re-read at 2 bytes/element.

MoE weight HBM traffic per decoded token (gpt-oss-20b: hidden = inter = 2880,
fc1 n = 5760, top_k = 4 → 99.5 M active weight elements/token; H200 ≈ 4.8 TB/s):

| Path | Weight bytes/token | Ideal time |
| --- | ---: | ---: |
| INT4 fused (read 4-bit once) | 49.8 MB | ~10 µs |
| FP4 fallback (dequant read 4-bit + write FP16 + GEMM read FP16) | 447.9 MB | ~93 µs |

That is a **9× weight-traffic penalty**. Two structural problems compound it:

1. **The dequant cost is batch-size-independent.** It touches the entire active
   weight matrix every token regardless of the (decode) batch of 1, so it can
   never be amortized — pure data movement that does no math.
2. **The dense FP16 grouped GEMM is a prefill-shaped kernel run on 4 rows.** At
   decode (expanded rows = num_rows × top_k = 4) it is latency/launch bound and
   badly under-utilizes the GPU, on top of reading 4× the weight bytes.

### Why llama.cpp reaches ~297 tps

llama.cpp uses a **fused MXFP4 W4A8** MoE kernel (`vec_dot_mxfp4_q8_1`): weights
stay 4-bit, are dequantized **in-register** to an int8 LUT, the activation is
quantized to int8 (Q8_1) and the inner product uses integer `dp4a`. No HBM
dequant round-trip, one kernel — architecturally the **same** strategy as ORT's
INT4 `fpA_intB` path. That is why llama.cpp FP4 (~297 tps) lands in the same
class as ORT INT4 (~270 tps), while ORT FP4 (~163 tps) pays the fallback's 9×
weight-traffic tax.

### Why the `ORT_ENABLE_FP4_GEMV` experiment did not close it

A fused MXFP4 W4A16 GEMV decode path was prototyped (gated behind
`ORT_ENABLE_FP4_GEMV`, default OFF). It engages on decode shapes and is
accuracy-correct, but gives **no e2e speedup** (163.0 vs 163.1 tps). It replaces
only the fc1/fc2 MoE matmuls with a 4-bit GEMV; the surrounding dense grouped
GEMM, dequant, and per-token launch overhead in the genai gpt-oss decode loop
still dominate, so the saved weight-read does not surface end-to-end at batch 1.

### Path to closing the gap

The decisive fix is a **single fused MXFP4 grouped GEMM** that keeps weights
4-bit in HBM and dequantizes e2m1 + e8m0 **in registers** inside the matmul
(the MXFP4 analogue of `fpA_intB`, or a W4A8 `dp4a` kernel like llama.cpp's),
eliminating both the HBM dequant round-trip and the dense-FP16 weight read. That
removes the ~9× weight-traffic penalty and should bring FP4 decode into the
INT4/llama.cpp class. Until such a kernel exists for sm_90, the dequant fallback
(now active-experts-only) is the correct default, and **FP4 remains advantageous
only on sm_120+** where the native MXFP4 tensor-core GEMM is available.

## Reproducing

Build ORT (optionally with FP4 kernels), build genai, export the model, and
verify with the orchestration script:

```bash
# INT4 QMoE (default)
./run_gpt_oss.sh --all

# FP4 (MXFP4) QMoE — builds ORT with onnxruntime_USE_FP4_QMOE=ON and
# exports with use_fp4_moe=true (requires CUDA Toolkit 12.8+)
./run_gpt_oss.sh --all --fp4
```

Benchmark across engines:

```bash
# INT4 model (CUDA graph on)
./bench_gpt_oss_compare.sh --all --prompt-len 512 --gen-len 128 --cuda-graph 1

# FP4 model (override the ORT model path)
ORT_MODEL=/home/tianlei/gptoss20b_fp4_qmoe \
  ./bench_gpt_oss_compare.sh --ort --prompt-len 512 --gen-len 128 --cuda-graph 1
```

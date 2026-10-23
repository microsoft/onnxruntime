# INT3 QMoE CUDA Correctness Baseline and Experimental Results

## Scope

This work extends the compatible QMoE v1 foundation in PR #33394 with actual
CUDA execution for effective FC1/FC2 widths `(3,3)`, `(3,4)`, and `(4,3)`.
It is a correctness baseline, not completion of the delivery plan's optimized
decode, private prepack, or GEMM-based prefill milestones.

### Delivery Plan Accounting

Against the [delivery plan at revision 2b1185e24a](https://github.com/microsoft/onnxruntime/blob/2b1185e24a5a79c56fd01226ad81b016450d0ef3/docs/design/int3-qmoe-end-to-end-delivery-plan.md):

| Proposed Stage | Delivered Here | Still Open |
| --- | --- | --- |
| PR2 / P2 | Independent references, bounded packed CUDA correctness, actual Session coverage | Clean provider-target build, formal integration tests, complete stage acceptance |
| PR3 / P3 | In-place extraction and FP32 accumulation for both FCs, three width pairs, repeated calls and memcheck | Versioned provider-private prepack, cache identity/lifetime, cached/runtime-scale qualification, measured prepack costs |
| PR4a / P4a | Functional bounded direct-packed prefill and full-model long-context runs | Required selected-expert/tiled dequantization plus existing GEMM, dedicated scratch/transition acceptance |
| PR4b/4c | No optimized native packed prefill delivery | Candidate kernel evidence, integration and performance acceptance |

Therefore this PR does **not** claim completion of PR2 through PR4 or the full
baseline exit gate. Full-model measurements cover `(3,3)` versus `(4,4)`;
`(3,4)` and `(4,3)` have small-fixture coverage, not full-model qualification.

The implementation supports FP16/BF16, block sizes 32/64/128, raw row-local
LSB-first packed weights, interleaved SwiGLU, biases, optional zero points,
existing top-k/supported sparse-mixer routing, separate aggregation weights,
and bounded token activation tiling. It does not materialize all experts as
floating-point weights. Legacy effective-INT2/INT4/INT8 branches are retained.

INT3 content validation checks finite nonnegative scales and symmetric zero
points/padding on every call and synchronizes the stream. Immutable initializer
validation caching and CUDA graph capture support are not implemented.

## Full Model Export

Experiments performed on 2026-10-21 used Qwen/Qwen3-30B-A3B revision
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`: 48 layers, hidden size 2048,
128 experts, top-k 8, and expert intermediate size 768.

Both matched exports use RTN, block64, FP16 computation/scales, INT4 attention
and embeddings, and an INT8 lm_head. All routed expert gate/up/down weights
are INT3 in the INT3 model and INT4 in the control; there is no selective-layer
INT2/INT4 substitute for INT3. Static audits found 48 QMoE nodes, 192 INT4
attention MatMulNBits nodes, and one INT8 head in each model. Every layer's
packed initializer shape and external-data byte range was checked.

| Model | ONNX Graph Plus External Weights (Bytes) |
| --- | ---: |
| All-expert INT3 | 12,782,047,512 |
| All-expert INT4 | 16,405,926,354 |

The total model-size reduction is 22.09%. Producer/export experiments depend
on local companion changes in Olive (`olive/common/quant/utils.py` and
`hf_utils.py`) and Mobius (`src/mobius/_weight_utils.py`); those changes are
not included in this ORT PR. The INT3 producer computes
`max(negative_peak/4, positive_peak/3)`, rounds scales to storage dtype before
nearest-even code selection, and uses unsigned offset 4. Nonzero scale
underflow is rejected.

## Build Provenance and Limitations

These experiments used isolated CUDA12.8 artifacts, cached core/Python bindings
from commit `98468bff47`, and a locally relinked CUDA provider. This is NOT a
clean build of this branch. Two cache-mixing ABI incompatibilities were
identified during full-model bring-up: external initializer loading interfaces
and `CutlassGemmConfig` layout. A build-only old-header overlay aligned the
new control objects with cached runner objects. The overlay is an experimental
build aid, not a source-level production fix and not part of this PR.

The provider SHA-256 used for both models was
`5f7af3b7da0454ed276dd46c9f808f6ca77b94621c42d4d6645662c90a80fdf9`.
Source and destination hashes were checked after container transfer.
Clean-build validation remains required before considering this ready to merge.

## Matched Performance Measurements

Hardware: NVIDIA A100-SXM4-80GB. Batch size 1, fixed 32 output tokens,
one warmup, three measured trials, scratch limit 67,108,864 bytes, no CUDA
graph capture. Synthetic repeated-context prompts use the same tokenizer and
template. EOS is ignored for fixed-length measurement. This is not a quality test.

TTFT excludes loading and KV reset and includes first-token transfer/argmax.
Prefill measures synchronized ORT execution with full-sequence logits. Decode
includes binding, GPU execution, last-row transfer, finite checks, and CPU argmax.
CPU shape operations remain allowed in these full-model sessions.

| Input Tokens | Expert Bits | TTFT p50 (ms) | Prefill (token/s) | Decode (token/s) | Sampled Process Peak (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 128 | 3 | 901.41 | 142.18 | 45.56 | 13.44 |
| 128 | 4 | 30.59 | 4399.88 | 114.39 | 31.78 |
| 512 | 3 | 3544.41 | 144.51 | 44.62 | 14.46 |
| 512 | 4 | 61.58 | 8541.29 | 108.80 | 31.80 |
| 2048 | 3 | 14099.40 | 145.27 | 42.75 | 14.45 |
| 2048 | 4 | 173.88 | 11914.46 | 100.36 | 31.80 |
| 4096 | 3 | 28184.65 | 145.34 | 40.69 | 16.46 |
| 4096 | 4 | 335.17 | 12287.25 | 90.14 | 34.81 |

Memory is NVML process residency, not exact allocator peak. Requested sampling
interval was 5ms; the INT3/128 run had a maximum sample gap of 53.84ms.
INT4 retains its existing prepack/workspace behavior; memory is a measured
whole-runtime difference, not a claim of an intrinsic bit-width-only reduction.

**INT3 is substantially slower than INT4, especially during prefill.** The
current direct-packed projection implementation does not satisfy the planned
selected-expert/tiled-dequant plus GEMM prefill milestone. No performance
acceptance or speedup is claimed.

## MMLU Pipeline Check

Protocol: lm-eval 0.4.13, MMLU, 5-shot, seed1234 for all random generators,
batch1, maximum length4096, no chat template, the first ONE test document per
subject (57 documents total). Both models used the same runtime and 64MiB
scratch configuration. Every document, prompt, and target hash matched.

| Model | Correct / Evaluated | Diagnostic Accuracy |
| --- | ---: | ---: |
| All-expert INT3 | 40 / 57 | 70.18% |
| All-expert INT4 | 49 / 57 | 85.96% |

This is a small-sample pipeline diagnostic, NOT a full MMLU quality estimate.
Full 57-subject evaluation was launched using eight subject-partitioned GPU
workers. No final full-MMLU summary was available when this report was prepared;
it is deliberately not reported as completed or substituted by the numbers above.

Machine-readable model audits, complete performance reports including trials,
and small-sample evaluation protocol/aggregates are in
[int3-qmoe-cuda-baseline-results.json](int3-qmoe-cuda-baseline-results.json).
Dataset question text and model weights are not included in this PR.

## Executed Validation

- Standalone packed-loader CUDA test: 80 cases, memcheck zero errors.
- Standalone packed double-projection CUDA test: 216 FP16/BF16 cases, three
  calls each, memcheck zero errors.
- Actual CUDA-only C API Session tests: 82 models, valid cases repeated three
  times; malformed scale/zero-point/padding rejection included. CPU fallback
  disabled. Two representative actual Sessions also passed memcheck.
- Python reference tests: 223 passed, one opt-in interoperability skip.
- Companion Mobius nearby tests: 170 passed; Olive quantization tests: 272 passed.
- Six Olive -> Mobius -> serialized ONNX -> actual CUDA Session cases cover
  both dtypes and three width pairs, nonzero weights, real gate/up interleave
  nodes, and independent bit-unpacking/NumPy projection references. Three
  repetitions passed in both C API and Python I/O-binding validation.
- Full INT3 and INT4 models completed prefill and KV-cached generation, all
  eight matched performance runs, and the paired 57-document evaluation above.

The new formal OpTester tests compile but have NOT been reliably executed:
cached test-helper ABI divergence prevented running that binary. Standalone
CUDA tests have their own main functions and are not integrated into the
standard unit-test target. C API harness coverage does not replace a clean
build and normal CI execution.

## Follow-Up Before Production Readiness

- Clean CUDA build, formal OpTester execution, and broader legacy regression.
- Versioned provider-private prepack and immutable initializer validation cache.
- Efficient decode and selected-expert/tiled-dequant plus GEMM prefill.
- Full matched MMLU results with sample-level protocol verification.
- Durable archive and clean-build reproduction of experiment artifacts; model
  files currently reside on temporary storage.
# QDQ vs MatMulNBits Benchmarking

Measures raw CPU inference latency comparing QDQ (DequantizeLinear+MatMul) and MatMulNBits representations across quantized Qwen2 models.

## Setup

```
pip install onnxruntime numpy
```

Models should be in a directory (default `C:/dev/llm`) with naming convention:
- `mnb-qwen-{size}-{bits}-{sym|asym}` (MatMulNBits)
- `qdq-qwen-{size}-{bits}-{block|channel}-{sym|asym}-{signed|unsigned}` (QDQ)

Each directory must contain `model.onnx`.

## Quick Start

```bash
# Single model benchmark
python benchmark.py -m C:/dev/llm/mnb-qwen-0.5b-4-sym/model.onnx

# Dry run to preview batch experiment commands
python run_experiments.py --preset validate --dry-run

# Run validation preset (0.5B 4-bit models, ~2 min)
python run_experiments.py --preset validate

# Run all 96 models
python run_experiments.py --preset full
```

## Scripts

### benchmark.py

Benchmarks a single ONNX model. Creates dummy LLM inputs (input_ids, attention_mask, empty past KV cache) and measures inference latency across sequence lengths.

```bash
python benchmark.py -m model.onnx                     # defaults: seq_lengths=[128,256,512,1024], 1 warmup, 1 iteration
python benchmark.py -m model.onnx -s 128 512 -w 3 -i 10  # custom seq lengths, 3 warmup, 10 iterations
python benchmark.py -m model_qdq.onnx --disable-qdq-fusion  # prevent DQ+MatMul -> MatMulNBits fusion
python benchmark.py -m model_2bit.onnx --enable-lut-gemm     # LUT GEMM for 2-bit models
python benchmark.py -m model.onnx --perf-test                # cross-check with onnxruntime_perf_test
python benchmark.py -m model.onnx --save-optimized-model opt.onnx  # save ORT-optimized graph for inspection
```

Output: JSON file in `results/` with latency statistics (mean, std, p50, p95, p99) per sequence length.

### run_experiments.py

Batch runner that discovers models, runs `benchmark.py` as a subprocess for each (sequential, one at a time), and aggregates results.

```bash
python run_experiments.py --preset validate                  # 0.5B 4-bit only, 3 iterations
python run_experiments.py --preset quick                     # 0.5B + 1.5B, 10 iterations
python run_experiments.py --preset full                      # all models, 10 iterations
python run_experiments.py --preset full --bits 4 8           # exclude 2-bit models
python run_experiments.py --preset full --model-sizes 0.5b   # filter by size
python run_experiments.py --aggregate-only                   # re-aggregate existing results
```

Key flags: `--model-dir`, `--results-dir`, `--format-types`, `--no-unfused`, `--dry-run`, `--timeout`

## Output Structure

```
results/
  experiment_config.json    # preset, filters, system info
  benchmark_results.csv     # full detail (all statistics per model/seq_length)
  summary.csv               # lightweight (format, size, bits, symmetry, scenario, mean_ms)
  failed_models.json        # failures (if any)
  per_model/
    mnb-qwen-0.5b-4-sym_native.json
    qdq-qwen-0.5b-4-block-sym-signed_qdq_fused.json
    ...
```

## Scenarios

| Model Format | Scenario | Session Options |
|---|---|---|
| mnb | `native` | None — MatMulNBits ops run directly |
| qdq | `qdq_fused` | None — ORT default optimizer fuses DequantizeLinear+MatMul into MatMulNBits |
| qdq | `qdq_unfused` | `disabled_optimizers=["QDQSelectorActionTransformer"]` — keeps DQ+MatMul separate |

For 2-bit models, `mlas.use_lut_gemm=1` is additionally enabled in all scenarios (auto-detected by run_experiments.py).

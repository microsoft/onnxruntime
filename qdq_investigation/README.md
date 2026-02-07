# ORT QDQ vs MatMulNBits Performance Investigation

This directory contains tools for benchmarking and comparing the performance of different quantization representations in ONNX Runtime on CPU EP:

1. **MatMulNBits** (QOperator format) - Native quantized operator
2. **DequantizeLinear → MatMul** (QDQ format) - Standard ONNX operator composition

## Files

- `benchmark_qdq_vs_matmulnbits.ipynb` - Main benchmark notebook
- `model_list.md` - List of models to generate for the investigation

## Models to Generate

You need to generate the following models for each base LLM:

### Base Models
- Qwen2.5-1.5B-Instruct (`Qwen/Qwen2.5-1.5B-Instruct`)
- Phi-3.5-mini-instruct (`microsoft/Phi-3.5-mini-instruct`)
- Llama-3.1-8B-Instruct (`meta-llama/Llama-3.1-8B-Instruct`)

### Quantization Configurations

| # | Format | Bits | Block Size | Symmetric | Model Suffix |
|---|--------|------|------------|-----------|--------------|
| 1 | QOperator (MatMulNBits) | 4 | 64 | Yes | `matmulnbits_4b_bs64_sym` |
| 2 | QOperator (MatMulNBits) | 4 | 64 | No | `matmulnbits_4b_bs64_asym` |
| 3 | QOperator (MatMulNBits) | 4 | 128 | Yes | `matmulnbits_4b_bs128_sym` |
| 4 | QOperator (MatMulNBits) | 4 | 128 | No | `matmulnbits_4b_bs128_asym` |
| 5 | QOperator (MatMulNBits) | 4 | -1 (per-channel) | Yes | `matmulnbits_4b_perchannel_sym` |
| 6 | QDQ | 4 | 64 | Yes | `qdq_4b_bs64_sym` |
| 7 | QDQ | 4 | 64 | No | `qdq_4b_bs64_asym` |
| 8 | QDQ | 4 | 128 | Yes | `qdq_4b_bs128_sym` |
| 9 | QDQ | 4 | 128 | No | `qdq_4b_bs128_asym` |
| 10 | QOperator (MatMulNBits) | 8 | 64 | Yes | `matmulnbits_8b_bs64_sym` |
| 11 | QDQ | 8 | 64 | Yes | `qdq_8b_bs64_sym` |

**Total: 33 models** (11 configurations × 3 base models)

## Disabling QDQ Fusion

To compare pure DequantizeLinear→MatMul performance (Scenario C), you need to disable the `DQMatMulToMatMulNBits` graph optimizer.

### Method 1: Using `disabled_optimizers` in onnxruntime

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"],
    disabled_optimizers=["DQMatMulToMatMulNBits"]
)
```

### Reference in ONNX Runtime Code

- **Python API**: `onnxruntime/python/onnxruntime_inference_collection.py` (lines 501-593)
- **Fusion Registration**: `onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.cc` (line 302)

## Metrics

- **TTFT (Time to First Token)**: Measures prompt processing latency
- **TPS (Tokens per Second)**: Measures decode throughput

## Usage

1. Generate all required models using onnxruntime-genai model builder
2. Update `MODEL_CONFIGS` in the notebook with your model paths
3. Run the benchmark notebook
4. Analyze results and export reports

## Test Scenarios

| Scenario | Description | Models Used |
|----------|-------------|-------------|
| A (native) | MatMulNBits models as-is | `*_matmulnbits_*` |
| B (qdq_fused) | QDQ models with auto-fusion to MatMulNBits | `*_qdq_*` |
| C (qdq_unfused) | QDQ models with fusion disabled | `*_qdq_*` (pre-processed or runtime disabled) |

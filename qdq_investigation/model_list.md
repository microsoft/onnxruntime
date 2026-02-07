# Model List for ORT QDQ Investigation

## Summary

| Base Model | HuggingFace ID | Parameters |
|------------|----------------|------------|
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B |
| Phi-3.5-mini-instruct | `microsoft/Phi-3.5-mini-instruct` | 3.8B |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8B |

## Full Model Matrix

### Qwen2.5-1.5B-Instruct

| Model Name | Format | Bits | Block Size | Symmetric |
|------------|--------|------|------------|-----------|
| `qwen_matmulnbits_4b_bs64_sym` | QOperator | 4 | 64 | ✓ |
| `qwen_matmulnbits_4b_bs64_asym` | QOperator | 4 | 64 | ✗ |
| `qwen_matmulnbits_4b_bs128_sym` | QOperator | 4 | 128 | ✓ |
| `qwen_matmulnbits_4b_bs128_asym` | QOperator | 4 | 128 | ✗ |
| `qwen_matmulnbits_4b_perchannel_sym` | QOperator | 4 | -1 | ✓ |
| `qwen_qdq_4b_bs64_sym` | QDQ | 4 | 64 | ✓ |
| `qwen_qdq_4b_bs64_asym` | QDQ | 4 | 64 | ✗ |
| `qwen_qdq_4b_bs128_sym` | QDQ | 4 | 128 | ✓ |
| `qwen_qdq_4b_bs128_asym` | QDQ | 4 | 128 | ✗ |
| `qwen_matmulnbits_8b_bs64_sym` | QOperator | 8 | 64 | ✓ |
| `qwen_qdq_8b_bs64_sym` | QDQ | 8 | 64 | ✓ |

### Phi-3.5-mini-instruct

| Model Name | Format | Bits | Block Size | Symmetric |
|------------|--------|------|------------|-----------|
| `phi_matmulnbits_4b_bs64_sym` | QOperator | 4 | 64 | ✓ |
| `phi_matmulnbits_4b_bs64_asym` | QOperator | 4 | 64 | ✗ |
| `phi_matmulnbits_4b_bs128_sym` | QOperator | 4 | 128 | ✓ |
| `phi_matmulnbits_4b_bs128_asym` | QOperator | 4 | 128 | ✗ |
| `phi_matmulnbits_4b_perchannel_sym` | QOperator | 4 | -1 | ✓ |
| `phi_qdq_4b_bs64_sym` | QDQ | 4 | 64 | ✓ |
| `phi_qdq_4b_bs64_asym` | QDQ | 4 | 64 | ✗ |
| `phi_qdq_4b_bs128_sym` | QDQ | 4 | 128 | ✓ |
| `phi_qdq_4b_bs128_asym` | QDQ | 4 | 128 | ✗ |
| `phi_matmulnbits_8b_bs64_sym` | QOperator | 8 | 64 | ✓ |
| `phi_qdq_8b_bs64_sym` | QDQ | 8 | 64 | ✓ |

### Llama-3.1-8B-Instruct

| Model Name | Format | Bits | Block Size | Symmetric |
|------------|--------|------|------------|-----------|
| `llama_matmulnbits_4b_bs64_sym` | QOperator | 4 | 64 | ✓ |
| `llama_matmulnbits_4b_bs64_asym` | QOperator | 4 | 64 | ✗ |
| `llama_matmulnbits_4b_bs128_sym` | QOperator | 4 | 128 | ✓ |
| `llama_matmulnbits_4b_bs128_asym` | QOperator | 4 | 128 | ✗ |
| `llama_matmulnbits_4b_perchannel_sym` | QOperator | 4 | -1 | ✓ |
| `llama_qdq_4b_bs64_sym` | QDQ | 4 | 64 | ✓ |
| `llama_qdq_4b_bs64_asym` | QDQ | 4 | 64 | ✗ |
| `llama_qdq_4b_bs128_sym` | QDQ | 4 | 128 | ✓ |
| `llama_qdq_4b_bs128_asym` | QDQ | 4 | 128 | ✗ |
| `llama_matmulnbits_8b_bs64_sym` | QOperator | 8 | 64 | ✓ |
| `llama_qdq_8b_bs64_sym` | QDQ | 8 | 64 | ✓ |

## Generation Commands

Use onnxruntime-genai model builder to generate models. Example commands:

### QOperator Format (MatMulNBits)

```bash
# 4-bit, block_size=64, symmetric
python -m onnxruntime_genai.models.builder \
    -m Qwen/Qwen2.5-1.5B-Instruct \
    -o ./models/qwen/matmulnbits_4b_bs64_sym \
    -p int4 \
    -e cpu \
    --extra_options block_size=64 is_symmetric=1

# 4-bit, block_size=64, asymmetric
python -m onnxruntime_genai.models.builder \
    -m Qwen/Qwen2.5-1.5B-Instruct \
    -o ./models/qwen/matmulnbits_4b_bs64_asym \
    -p int4 \
    -e cpu \
    --extra_options block_size=64 is_symmetric=0
```

### QDQ Format

For QDQ format, you may need to use the onnxruntime quantization tools directly:

```python
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    DefaultWeightOnlyQuantConfig,
)
from onnxruntime.quantization.quant_utils import QuantFormat
import onnx

model = onnx.load("fp32_model.onnx")

config = DefaultWeightOnlyQuantConfig(
    block_size=64,
    is_symmetric=True,
    quant_format=QuantFormat.QDQ,  # Key: Use QDQ format
    bits=4,
)

quant = MatMulNBitsQuantizer(
    model=model,
    bits=4,
    algo_config=config,
)
quant.process()
quant.model.save_model_to_file("qdq_model.onnx", use_external_data_format=True)
```

## Recommended Directory Structure

```
models/
├── qwen2.5-1.5b/
│   ├── matmulnbits_4b_bs64_sym/
│   ├── matmulnbits_4b_bs64_asym/
│   ├── matmulnbits_4b_bs128_sym/
│   ├── qdq_4b_bs64_sym/
│   ├── qdq_4b_bs64_asym/
│   └── ...
├── phi-3.5-mini/
│   └── ...
└── llama-3.1-8b/
    └── ...
```

# Neutron Execution Provider

Execution provider for NXP Neutron NPUs.

## Overview

The Neutron EP accelerates quantized matrix operations on NXP i.MX series processors with Neutron NPU.
It supports common operators used in large language models (LLMs) with 4-bit/8-bit weight quantization.

## Supported Operators

| Operator | Domain | Type Support |
|----------|--------|--------------|
| MatMulNBits | com.microsoft | float, uint8_t, float/MLFloat16/int32_t |
| MatMulInteger | ONNX | uint8_t, int8_t |
| MatMulIntegerToFloat | com.microsoft | uint8_t, int8_t |
| QLinearMatMul | ONNX | uint8_t, int8_t |
| QGemm | com.microsoft | uint8_t, int8_t |
| NeutronGraph | neutron | int8_t |

## Offline Prepacking

To reduce model loading time, prepack weights offline:

```bash
cd onnxruntime/core/providers/neutron/tools
python3 neutron_offline_prepack.py -i model.onnx -o model_neutron.onnx
```

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input ONNX model | Required |
| `-o, --output` | Output model name | `{input}_neutron.onnx` |
| `-j, --jobs` | Parallel processes | CPU count |

### Online vs Offline Prepack

| Aspect | Online | Offline |
|--------|--------|---------|
| When | Model load time | Model conversion time |
| Memory peak | Higher (original + packed) | Lower |
| Load speed | Slower | Faster |

Requires `NEUTRON_AARCH64` definition for Neutron driver integration.

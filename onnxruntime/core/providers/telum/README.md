# Telum Execution Provider for ONNX Runtime

## Overview

The Telum Execution Provider (EP) enables hardware acceleration of neural network operations on IBM z16 (Telum) processors using the zDNN library and Neural Network Processing Assist (NNPA) facility.

## Key Features

- **Hardware Acceleration**: Leverages IBM z16 NNPA for accelerated inference
- **Transformer Optimized**: Designed specifically for transformer model inference
- **Static Shapes**: Requires compile-time known shapes for optimal performance
- **Synchronous Execution**: Deterministic, predictable execution model
- **Strict Validation**: Explicit error reporting instead of silent fallbacks

## Architecture

```
┌─────────────────────────────────────────┐
│         ONNX Runtime Core               │
├─────────────────────────────────────────┤
│      Telum Execution Provider           │
│  ┌───────────────────────────────────┐  │
│  │  Graph Transformers (Fusion)      │  │
│  ├───────────────────────────────────┤  │
│  │  Kernel Registry                  │  │
│  ├───────────────────────────────────┤  │
│  │  Tensor Converter                 │  │
│  ├───────────────────────────────────┤  │
│  │  Telum Allocator (4K aligned)     │  │
│  └───────────────────────────────────┘  │
├─────────────────────────────────────────┤
│            zDNN Library                 │
├─────────────────────────────────────────┤
│     IBM z16 NNPA Hardware               │
└─────────────────────────────────────────┘
```

## Supported Operations

### Implemented (Today)

| Operation | zDNN API | Constraints |
|-----------|----------|-------------|
| MatMul | `zdnn_matmul_op` / `zdnn_matmul_bcast_op` | Static shapes; limited broadcast patterns; explicit zero-bias `input_c` |
| Gemm | `zdnn_matmul_op` / `zdnn_matmul_transpose_op` | Static shapes; A/B must be 2D; bias fused only for a safe subset |
| Add/Sub/Mul/Div/Min/Max | `zdnn_add/sub/mul/div/min/max` | No broadcasting (shapes must match); rank <= 4 |
| Relu | `zdnn_relu` | Elementwise; rank <= 4 |
| Softmax | `zdnn_softmax` | Static shapes; axis == last dim only (coerced to `ZDNN_3DS`) |
| Gelu | `zdnn_gelu` | Elementwise; rank <= 4 |
| Tanh/Sigmoid/Exp/Log/Sqrt | `zdnn_tanh/sigmoid/exp/log/sqrt` | Elementwise; rank <= 4 |
| LayerNormalization | `zdnn_moments` + `zdnn_layernorm` | Static shapes; axis == last dim only; scale/bias shape [C]; scale/bias applied on CPU |

### Planned / In Progress

See `docs/telum/Telum_EP_TODO.md` for the full roadmap (broadcast patterns, prepacking/caching, and broader model coverage).

## Data Type Support

| ONNX Type | zDNN Type | Status |
|-----------|-----------|--------|
| FLOAT32 | FP32 | ✅ Supported |
| FLOAT16 | FP16 | ✅ Supported |
| BFLOAT16 | BFLOAT | ✅ Supported |

## Building

### Prerequisites

- IBM z16 or later processor
- zDNN library installed
- CMake
- C++17 compatible compiler

### Build Commands

```bash
# Using build.sh/build.py (recommended)
./build.sh --config Release --update --build --parallel \
  --use_telum --telum_home=/path/to/zdnn

# Using CMake directly
cmake -S cmake -B build \
  -Donnxruntime_USE_TELUM=ON \
  -DZDNN_ROOT=/path/to/zdnn \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Usage

### C API (Generic Provider Append)

Telum is exposed via the generic string-based EP append API:

```c
// OrtApis::SessionOptionsAppendExecutionProvider(...)
//
// provider_name:
//   - "TelumExecutionProvider" (canonical)
//   - "Telum"                 (short name)
//
// provider_options:
//   key/value string pairs, e.g. "log_fallbacks" -> "1"
```

There is currently no `Ort::SessionOptions::AppendExecutionProvider_Telum()` convenience wrapper; use the generic append entry point (or the Python provider list).

### Python API

```python
import onnxruntime as ort

# Create session with Telum EP
providers = [
    ('TelumExecutionProvider', {
        'strict_mode': True,
        'enable_fusion': True,
        'max_batch_size': 32
    }),
    'CPUExecutionProvider'  # Fallback
]

session = ort.InferenceSession('model.onnx', providers=providers)

# Run inference
outputs = session.run(None, inputs)
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `strict_mode` | bool | true | Reject unsupported ops instead of silent fallback |
| `enable_fusion` | bool | true | Enable operator fusion optimizations |
| `log_fallbacks` | bool | true | Log fallback decisions for debugging |
| `max_batch_size` | int | 32 | Maximum batch size for validation |
| `max_sequence_length` | int | 512 | Maximum sequence length for transformers |
| `create_arena` | bool | true | Use arena allocator |

## Constraints and Limitations

### Shape Requirements

- **Static Shapes Only**: All tensor dimensions must be known at session load time
- **No Dynamic Reshaping**: Runtime shape changes are not supported
- **Bounded Dimensions**: Dimensions must not exceed hardware limits

### Operation Constraints

- **Limited Broadcasting**: zDNN elementwise ops do not broadcast. Telum currently requires equal shapes for elementwise ops.
- **Transpose**: Gemm supports `transA/transB` attributes via zDNN transpose matmul. The standalone ONNX `Transpose` op is not currently offloaded.
- **Fixed Layouts**: Some operations require specific data layouts

### Performance Considerations

- **Memory Alignment**: All buffers are 4K-aligned for optimal performance
- **Synchronous Execution**: No async operations or background threads
- **Deterministic**: Same inputs always produce same outputs

## Troubleshooting

### Common Issues

#### 1. "zDNN/NNPA is not available"

**Cause**: Running on non-z16 hardware or NNPA not enabled

**Solution**: Verify you're on IBM z16 or later and NNPA is available:
```bash
cat /proc/cpuinfo | grep nnpa
```

#### 2. "Dynamic shapes not supported"

**Cause**: Model contains dynamic dimensions

**Solution**:
- Use fixed batch size and sequence length
- Reshape model to use static shapes
- Set `strict_mode=false` to allow CPU fallback

#### 3. "Operator not supported"

**Cause**: Operation not in supported list

**Solution**:
- Check supported operations list above
- Set `strict_mode=false` for CPU fallback
- Consider model modifications

## Tests

When built with `onnxruntime_USE_TELUM=ON`, a dedicated test binary is built:

- `onnxruntime_telum_test`

The Telum tests intentionally disable CPU EP fallback by default to ensure they actually execute Telum kernels (and fail if partitioning fell back to CPU).

### Debug Logging

Enable detailed logging:

```cpp
session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
```

Or set environment variable:
```bash
export ORT_LOG_SEVERITY_LEVEL=1
```

## Performance Tips

1. **Use Static Shapes**: Ensure all dimensions are compile-time constants
2. **Enable Fusion**: Keep `enable_fusion=true` for optimal performance
3. **Batch Processing**: Use larger batch sizes when possible
4. **Memory Reuse**: Reuse sessions across multiple inferences
5. **Profile First**: Use ORT profiling to identify bottlenecks

## Examples

See the `examples/` directory for complete examples:

- `bert_inference.cpp` - BERT model inference
- `gpt2_inference.cpp` - GPT-2 model inference
- `custom_model.cpp` - Custom transformer model

## Testing

Run the test suite:

```bash
cd build
ctest -R telum -V
```

## Contributing

See [CONTRIBUTING.md](../../../../../../CONTRIBUTING.md) for guidelines.

## License

Copyright (c) Microsoft Corporation. Licensed under the MIT License.

## References

- [zDNN Documentation](https://github.com/IBM/zDNN)
- [IBM z16 Technical Guide](https://www.ibm.com/docs/en/z16)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)

## Support

For issues and questions:
- GitHub Issues: https://github.com/microsoft/onnxruntime/issues
- Tag: `telum-ep` or `zdnn`

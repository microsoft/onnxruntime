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

### Priority 0 (Critical for Transformers)

| Operation | zDNN API | Constraints |
|-----------|----------|-------------|
| MatMul | `zdnn_matmul_op` | Static shapes, aligned dims |
| Gemm | `zdnn_matmul_op` | Static shapes, optional bias |
| Add | `zdnn_add` | Limited broadcast |
| Relu | `zdnn_relu` | Elementwise |
| Gelu | `zdnn_gelu` | Elementwise |
| Softmax | `zdnn_softmax` | Last-dim only |
| LayerNormalization | `zdnn_layernorm` | Fixed hidden size |

### Priority 1 (Important Operations)

| Operation | zDNN API | Constraints |
|-----------|----------|-------------|
| Sub | `zdnn_sub` | Same as Add |
| Mul | `zdnn_mul` | Same-shape only |
| Tanh | `zdnn_tanh` | Elementwise |
| Sigmoid | `zdnn_sigmoid` | Elementwise |
| Exp | `zdnn_exp` | Elementwise |
| Log | `zdnn_log` | Elementwise |
| Sqrt | `zdnn_sqrt` | Elementwise |
| Min | `zdnn_min` | Elementwise |
| Max | `zdnn_max` | Elementwise |
| Div | `zdnn_div` | Elementwise |

## Data Type Support

| ONNX Type | zDNN Type | Status |
|-----------|-----------|--------|
| FLOAT32 | FP32 | ✅ Supported |
| FLOAT16 | FP16 | ✅ Supported |
| BFLOAT16 | BFLOAT | ✅ Supported |
| INT8 | INT8 | ⚠️ Quantized ops only |
| INT32 | INT32 | ⚠️ Quantized ops only |

## Building

### Prerequisites

- IBM z16 or later processor
- zDNN library installed
- CMake 3.18 or later
- C++17 compatible compiler

### Build Commands

```bash
# Configure ONNX Runtime with Telum support
cd onnxruntime
./build.sh --config Release \
           --use_telum \
           --telum_home=/path/to/zdnn \
           --parallel

# Or with CMake directly
mkdir build && cd build
cmake -DONNXRUNTIME_USE_TELUM=ON \
      -DZDNN_ROOT=/path/to/zdnn \
      -DCMAKE_BUILD_TYPE=Release \
      ..
make -j$(nproc)
```

## Usage

### C++ API

```cpp
#include <onnxruntime_cxx_api.h>

// Create session options
Ort::SessionOptions session_options;

// Add Telum EP with default settings
session_options.AppendExecutionProvider_Telum();

// Or with custom settings
TelumExecutionProviderInfo telum_info;
telum_info.strict_mode = true;
telum_info.enable_fusion = true;
telum_info.max_batch_size = 32;
session_options.AppendExecutionProvider("Telum", telum_info);

// Create session
Ort::Session session(env, "model.onnx", session_options);

// Run inference as usual
auto output = session.Run(...);
```

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

- **Limited Broadcasting**: Only specific broadcast patterns are supported
- **No Transpose**: Transpose operations must be fused or handled upstream
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

# GitHub Copilot Instructions for ONNX Runtime

## Project Overview

ONNX Runtime is a cross-platform inference and training machine-learning accelerator developed by Microsoft. It supports models from deep learning frameworks like PyTorch and TensorFlow/Keras, as well as classical ML libraries like scikit-learn, LightGBM, and XGBoost.

### Key Architecture Components

- **Core Runtime** (`onnxruntime/core/`): Main inference engine and framework
- **Execution Providers** (`onnxruntime/core/providers/`): Hardware-specific optimizations (CPU, CUDA, DirectML, etc.)
- **Graph Optimization** (`onnxruntime/core/optimizer/`): Model optimization and transformation
- **Session Management** (`onnxruntime/core/session/`): Model loading and execution lifecycle
- **Language Bindings**: C# (`csharp/`), Python (`python/`), Java (`java/`), JavaScript (`js/`), Objective-C (`objectivec/`), Rust (`rust/`)

## C++ Coding Standards

Follow Google C++ Style Guide with these ONNX Runtime specific modifications:

### Line Length and Formatting
```cpp
// Max line length: 120 characters (aim for 80)
// Use .clang-format in root directory
```

### Memory Management and Containers
```cpp
// Prefer ONNX Runtime optimized containers to reduce allocations:
#include "core/common/inlined_containers.h"

// Use these instead of std:: equivalents:
InlinedVector<int64_t> values;           // Instead of std::vector
InlinedHashMap<string, int> map;         // Instead of std::unordered_map
InlinedHashSet<string> set;              // Instead of std::unordered_set
TensorShapeVector shape;                 // For tensor shapes (small buffer optimized)

// For containers requiring pointer stability:
NodeHashMap<string, Value> stable_map;
NodeHashSet<string> stable_set;

// Always reserve capacity when size is known:
values.reserve(expected_size);
```

### Function Parameters
```cpp
// Prefer gsl::span for contiguous container inputs:
void ProcessValues(gsl::span<const int64_t> values);  // Instead of const std::vector<int64_t>&
void ProcessNodes(gsl::span<const Node* const> nodes); // For pointer arrays

// Prefer std::string_view for string inputs:
void ProcessName(std::string_view name);  // Instead of const std::string&

// Use AsSpan() for initializer lists:
ProcessValues(AsSpan<int64_t>({1, 2, 3}));
```

### Smart Pointers and RAII
```cpp
// Prefer std::unique_ptr and std::make_unique:
auto session = std::make_unique<InferenceSession>(options);

// Avoid std::shared_ptr unless ownership is truly shared
// Use std::optional instead of std::unique_ptr for optional members:
class Config {
  std::optional<ModelMetadata> metadata_;  // Instead of std::unique_ptr<ModelMetadata>
};
```

### Error Handling
```cpp
#include "core/common/status.h"

// Use Status for error handling:
onnxruntime::common::Status ProcessModel(const Model& model) {
  if (!model.IsValid()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid model provided");
  }
  // ... processing
  return Status::OK();
}

// Check status in calling code:
ORT_RETURN_IF_ERROR(ProcessModel(model));
```

### Memory Safety
```cpp
#include "core/common/safeint.h"

// Use SafeInt for memory size calculations:
SafeInt<size_t> total_size = element_count * element_size;
auto buffer = std::make_unique<uint8_t[]>(total_size);
```

### Class Design
```cpp
class MyClass {
 public:
  // Disable copy/move initially, enable selectively when needed
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MyClass);

  // Qualify auto with const, *, &, && as appropriate
  auto GetValue() const -> const Value&;
  auto GetMutableValue() -> Value&;
};
```

### Control Flow
```cpp
// Don't use else after return:
Status ValidateInput(const Tensor& input) {
  if (input.Shape().empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Empty shape");
  }
  if (input.DataType() != DataTypeImpl::GetType<float>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expected float type");
  }
  // Continue processing...
  return Status::OK();
}
```

### Namespace Usage
```cpp
// Limited scope using namespace is allowed:
namespace onnxruntime {
void SomeFunction() {
  using namespace common;  // OK in function scope
  Status status = ProcessData();
}
}  // namespace onnxruntime

// Never use "using namespace" in headers at global scope
```

## Python Coding Standards

- Follow Black formatter with 120 character line length
- Adhere to PEP8 and Google Python Style Guide
- Use type hints and pyright for static type checking
- Use unittest framework for testing, pytest for running tests

```python
from typing import Optional, List, Union
import onnxruntime as ort

def create_session(model_path: str, providers: Optional[List[str]] = None) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session.

    Args:
        model_path: Path to the ONNX model file.
        providers: List of execution providers to use.

    Returns:
        Configured inference session.
    """
    if providers is None:
        providers = ['CPUExecutionProvider']

    return ort.InferenceSession(model_path, providers=providers)
```

## Common Patterns and Anti-Patterns

### Execution Provider Pattern
```cpp
class MyExecutionProvider : public IExecutionProvider {
 public:
  MyExecutionProvider(const MyExecutionProviderInfo& info);

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const IKernelLookup& kernel_lookup) const override;

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;
};
```

### Kernel Implementation Pattern
```cpp
template<typename T>
class MyKernel final : public OpKernel {
 public:
  MyKernel(const OpKernelInfo& info) : OpKernel(info) {
    // Parse attributes
  }

  Status Compute(OpKernelContext* context) const override;
};
```

### Graph Transformation Pattern
```cpp
class MyGraphTransformer : public GraphTransformer {
 public:
  MyGraphTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers)
      : GraphTransformer("MyTransformer", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;
};
```

## Testing Guidelines

### C++ Testing
```cpp
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

TEST(MyOperatorTest, BasicFunctionality) {
  OpTester test("MyOperator", 1, kMyDomain);

  // Add inputs
  test.AddInput<float>("X", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

  // Add expected outputs
  test.AddOutput<float>("Y", {2, 3}, {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f});

  test.Run();
}
```

### Python Testing
```python
import unittest
import numpy as np
import onnxruntime as ort

class TestMyFeature(unittest.TestCase):
    def test_basic_inference(self):
        """Test basic inference functionality."""
        # Test implementation
        pass

    def test_error_handling(self):
        """Test error handling behavior."""
        with self.assertRaises(ValueError):
            # Code that should raise ValueError
            pass
```

## Build and Development

### CMake Patterns
```cmake
# Use onnxruntime_add_static_library for internal libraries
onnxruntime_add_static_library(my_component ${my_component_srcs})

# Link against common ONNX Runtime libraries
target_link_libraries(my_component PRIVATE
    onnxruntime_common
    onnxruntime_framework
)

# Use proper include directories
target_include_directories(my_component PRIVATE
    ${ONNXRUNTIME_ROOT}
    ${CMAKE_CURRENT_BINARY_DIR}
)
```

### Cross-Platform Considerations
```cpp
#ifdef _WIN32
  // Windows-specific code
#elif defined(__linux__)
  // Linux-specific code
#elif defined(__APPLE__)
  // macOS-specific code
#endif

// Use ORTCHAR_T for file paths:
Status LoadModel(const ORTCHAR_T* model_path);
```

## Documentation Standards

### Header Documentation
```cpp
/**
 * @brief Brief description of the class/function.
 *
 * Detailed description explaining the purpose, behavior, and any important
 * implementation details.
 *
 * @param param_name Description of parameter
 * @return Description of return value
 * @throws ExceptionType Description of when exception is thrown
 */
```

### Code Comments
```cpp
// Use single-line comments for brief explanations
// that don't require formal documentation

/*
 * Use multi-line comments for longer explanations
 * that span multiple lines but aren't formal documentation
 */
```

## Performance Considerations

- Minimize dynamic allocations using inlined containers
- Use `reserve()` instead of `resize()` when possible
- Prefer stack allocation and RAII
- Consider memory alignment for SIMD operations
- Profile before optimizing, measure impact of changes

## Key Files and Directories to Understand

- `include/onnxruntime/core/`: Public C++ API headers
- `onnxruntime/core/framework/`: Core framework classes (Tensor, MLValue, etc.)
- `onnxruntime/core/graph/`: Graph representation and manipulation
- `onnxruntime/core/session/`: Session management and inference execution
- `onnxruntime/core/providers/`: Execution provider implementations
- `onnxruntime/core/optimizer/`: Graph optimization passes
- `docs/`: Technical documentation and design documents

## Additional Guidelines

- Always check return values and handle errors appropriately
- Write unit tests for new functionality (aim for 80%+ coverage)
- Follow the contribution guidelines in CONTRIBUTING.md
- Use lintrunner for code formatting and linting
- Consider backward compatibility when modifying public APIs
- Document breaking changes and migration paths

# ONNX Runtime Kernel Registration Guide

This document explains how to register custom operator kernels in ONNX Runtime, focusing on the C++ implementation. Understanding kernel registration is essential when adding new operators, extending existing operators with new data types, or supporting new opset versions.

## Table of Contents
- [Overview](#overview)
- [Kernel Registry Architecture](#kernel-registry-architecture)
- [Kernel Registration Macros](#kernel-registration-macros)
- [Registration Steps](#registration-steps)
- [Complete Example: MatMulInteger](#complete-example-matmulinteger)
- [OpSet Versioning](#opset-versioning)
- [Best Practices](#best-practices)

## Overview

Kernel registration in ONNX Runtime connects operator definitions with their implementations (kernels) for specific execution providers. The registration process involves:

1. **Forward declaration** of the kernel class in the execution provider file
2. **Registration entry** using `BuildKernelCreateInfo` in the execution provider
3. **Kernel implementation** with the appropriate macro definition

The registration system uses a template-based approach with macros to generate kernel metadata and factory functions.

## Kernel Registry Architecture

ONNX Runtime uses a multi-layered registry architecture to manage kernel registrations. Understanding this architecture is essential for adding custom operators or understanding how kernels are discovered at runtime.

### Overview of Registry Classes

The kernel registry system consists of three main components:

1. **KernelRegistry**: Stores kernel definitions and factory functions
2. **KernelRegistryManager**: Manages multiple KernelRegistry instances with prioritization
3. **CustomRegistry**: User-facing class for registering custom kernels and schemas

### KernelRegistry

A `KernelRegistry` stores the mapping between operator definitions and their kernel implementations. Key characteristics:

- Stores `KernelCreateInfo` objects containing `KernelDef` and factory functions
- Provides lookup methods to find kernels matching a given operator node
- Typically each execution provider has one kernel registry
- Defined in `onnxruntime/core/framework/kernel_registry.h`

### KernelRegistryManager

Each `InferenceSession` has a `KernelRegistryManager` that manages multiple `KernelRegistry` instances from different sources. The manager is responsible for:

- Collecting registries from all registered execution providers
- Collecting registries from user-registered `CustomRegistry` instances
- Searching registries in priority order to find appropriate kernels
- Maintaining the mapping between execution provider types and their registries

**Priority Order** (highest to lowest):
1. User-registered `CustomRegistry` instances (last registered has highest priority)
2. Execution provider registries (in the order providers were registered)

This priority system allows users to override built-in kernels with custom implementations.

### How Execution Providers Register Kernels

Execution providers register their kernels by overriding the `GetKernelRegistry()` virtual method:

```cpp
class MyExecutionProvider : public IExecutionProvider {
  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    return my_kernel_registry_;
  }
};
```

During `InferenceSession::Initialize()`, the session calls `KernelRegistryManager::RegisterKernels(execution_providers)`, which:
1. Iterates through all registered execution providers
2. Calls `provider->GetKernelRegistry()` on each
3. Stores the registry with a mapping from provider type to registry

### CustomRegistry

The `CustomRegistry` class (defined in `include/onnxruntime/core/framework/customregistry.h`) is the **user-facing API** for registering custom operators. A `CustomRegistry` contains:

- A `KernelRegistry` for storing custom kernel definitions
- An `OnnxRuntimeOpSchemaRegistry` for storing custom operator schemas (in non-minimal builds)

#### Design Intent and Limitations

`CustomRegistry` was originally designed to support an immutable `onnxruntime.dll` that could be extended with new operator support at runtime without recompilation. This would allow:
- Shipping a stable core runtime
- Adding new operators via loadable modules
- Third-party operator extensions

While this vision hasn't been fully realized in its original form, the feature remains useful for runtime extensibility.

**Important Limitation - Schema and DLL Boundaries:**

When registering custom operators that need new schemas (not just custom kernels for existing ONNX ops), there is a fundamental challenge: ONNX operator schemas (`onnx::OpSchema`) are **not just data** - they contain **C++ function objects** for type and shape inference:

```cpp
// OpSchema contains std::function with C++ lambda/code
using InferenceFunction = std::function<void(InferenceContext&)>;

schema.TypeAndShapeInferenceFunction([](onnx::InferenceContext& ctx) {
    // C++ code that accesses protobuf structures
    const TypeProto* input_type = ctx.getInputType(0);
    // ... complex inference logic ...
});
```

These inference functions:
- Contain compiled C++ code with function pointers
- Access protobuf structures (`TypeProto`, `TensorProto`, etc.)
- Cannot safely cross DLL boundaries if the DLLs use different compilers, CRT versions, or protobuf versions

**Workaround for DLL Boundaries:**

The DirectML execution provider solves this by:
1. Providing a **COM-based ABI** (`IMLOperatorTypeInferrer`, `IMLOperatorShapeInferrer`) that uses virtual functions instead of `std::function`
2. Converting ABI calls back to ONNX `OpSchema` **within the same DLL** (onnxruntime.dll)
3. Wrapping the COM interface pointers in lambdas that can be stored in `OpSchema`

This allows external code to provide inference logic via stable COM interfaces, while keeping the actual `OpSchema` objects (with their C++ internals) contained within onnxruntime.dll.

**For User-Facing CustomRegistry:**

If you're using `CustomRegistry` within the same process/DLL as ONNX Runtime:
- ✅ You can create `OpSchema` objects directly with C++ lambdas
- ✅ Full access to ONNX schema features

If you need to register operators from a separate DLL:
- ⚠️ Passing `OpSchema` objects across DLL boundaries is unsafe
- ⚠️ You'll need an ABI layer (like DirectML's COM interfaces) or compile everything together
- ✅ Alternatively, just register kernels for existing ONNX ops (no schema needed)

#### Current Usage

**User-Facing Usage**: Developers can register custom operators for immediate use:

```cpp
#include "core/framework/customregistry.h"

// Create a custom registry
std::shared_ptr<CustomRegistry> registry = std::make_shared<CustomRegistry>();

// Register a custom kernel
KernelDefBuilder def;
def.SetName("MyOp")
    .SetDomain("com.mycompany")
    .SinceVersion(1)
    .Provider(kCpuExecutionProvider)
    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>());

auto create_kernel = [](FuncManager&, const OpKernelInfo& info, 
                       std::unique_ptr<OpKernel>& out) -> Status {
  out = std::make_unique<MyOpKernel>(info);
  return Status::OK();
};

registry->RegisterCustomKernel(def, create_kernel);

// Register custom schema (optional, for completely new operators)
std::vector<ONNX_NAMESPACE::OpSchema> schemas = {MyOpSchema()};
registry->RegisterOpSet(schemas, "com.mycompany", 1, 1000);

// Register with session
InferenceSession session(session_options, env);
session.RegisterCustomRegistry(registry);
```

**Internal Usage**: Some execution providers use `CustomRegistry` internally as an implementation detail. For example, the DirectML execution provider uses it to manage kernel registrations from its ABI boundary, but exposes the kernels via the standard `GetKernelRegistry()` method rather than `RegisterCustomRegistry()`.

#### Key Features

- **Override built-in operators**: Custom kernels for existing ONNX operators take precedence
- **Add new operators**: Register completely new operators with custom schemas
- **Multiple registries**: Register multiple `CustomRegistry` instances with different priorities
- **Schema validation**: Register operator schemas for model validation (non-minimal builds)

#### Examples

See `onnxruntime/test/framework/local_kernel_registry_test.cc` for comprehensive examples:
- Registering custom implementations of existing ONNX operators (e.g., custom `Mul` that actually does `Add`)
- Registering completely new operators with custom schemas (e.g., `Foo` operator)
- Handling operators with optional inputs and outputs

### Integration Flow

Here's how the registry system integrates with `InferenceSession`:

```
InferenceSession::Initialize()
  │
  ├─> kernel_registry_manager_.RegisterKernels(execution_providers_)
  │     └─> For each ExecutionProvider:
  │           registry = provider->GetKernelRegistry()
  │           Store in provider_type_to_registry_
  │
  ├─> [User may have called session.RegisterCustomRegistry(custom_reg)]
  │     └─> kernel_registry_manager_.RegisterKernelRegistry(custom_reg->GetKernelRegistry())
  │
  └─> Partition graph and create kernels
        └─> KernelRegistryManager searches registries in priority order
              1. Custom registries (user-registered, last-in-first)
              2. Execution provider registries
```

### Best Practices

When working with the registry system:

1. **Register before Initialize**: All registries (execution providers and custom registries) must be registered before calling `InferenceSession::Initialize()`
2. **Priority matters**: Register custom registries in reverse priority order (last registered = highest priority)
3. **Provider matching**: Ensure your custom kernels specify the correct execution provider
4. **Schema registration**: For new operators, register both the kernel and the schema for proper validation
5. **Testing**: Use the test utilities in `local_kernel_registry_test.cc` as examples for testing custom operators

## Kernel Registration Macros

ONNX Runtime provides several macros for kernel registration, each serving different use cases. All macros are defined in `include/onnxruntime/core/framework/op_kernel.h`.

### Basic Macros

#### 1. `ONNX_OPERATOR_KERNEL_CLASS_NAME`
Generates a unique class name for a kernel.

```cpp
ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)
```

**Expands to:** `provider##_##name##_##domain##_ver##ver`

**Example:**
```cpp
ONNX_OPERATOR_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)
// Expands to: kCpuExecutionProvider_MemcpyFromHost_kOnnxDomain_ver1
```

#### 2. `ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME`
Generates a class name for a kernel supporting a version range.

```cpp
ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name)
```

**Expands to:** `provider##_##name##_##domain##_ver##startver##_##endver`

**Example:**
```cpp
ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, Relu)
// Expands to: kCpuExecutionProvider_Relu_kOnnxDomain_ver6_12
```

#### 3. `ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME`
Generates a class name for a type-specific kernel.

```cpp
ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)
```

**Expands to:** `provider##_##name##_##domain##_ver##ver##_##type`

**Example:**
```cpp
ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger)
// Expands to: kCudaExecutionProvider_MatMulInteger_kOnnxDomain_ver10_int8_t
```

#### 4. `ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME`
Generates a class name for a type-specific kernel with version range.

```cpp
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name)
```

### Kernel Definition Macros

These macros define the actual kernel class and create the registration metadata.

#### 1. `ONNX_OPERATOR_KERNEL_EX`
Defines a kernel for a single opset version.

```cpp
ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, kernel_class)
```

**Parameters:**
- `name`: Operator name (e.g., `MatMul`)
- `domain`: Domain (e.g., `kOnnxDomain`, `kMSDomain`, `kMLDomain`)
- `ver`: Opset version number (single version)
- `provider`: Execution provider (e.g., `kCpuExecutionProvider`, `kCudaExecutionProvider`)
- `builder`: KernelDefBuilder expression for additional constraints
- `kernel_class`: The actual kernel class implementation

**Example:**
```cpp
ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorAndSequenceTensorTypesIRv9()),
    Memcpy);
```

#### 2. `ONNX_OPERATOR_VERSIONED_KERNEL_EX`
Defines a kernel for a range of opset versions.

```cpp
ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, startver, endver, provider, builder, kernel_class)
```

**Parameters:**
- `startver`: Starting opset version (inclusive)
- `endver`: Ending opset version (inclusive)
- Other parameters same as `ONNX_OPERATOR_KERNEL_EX`

**Example:**
```cpp
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Relu,
    kOnnxDomain,
    6,
    12,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Relu<float>);
```

#### 3. `ONNX_OPERATOR_TYPED_KERNEL_EX`
Defines a type-specific kernel for a single opset version.

```cpp
ONNX_OPERATOR_TYPED_KERNEL_EX(name, domain, ver, type, provider, builder, kernel_class)
```

**Parameters:**
- `type`: Data type (e.g., `float`, `int8_t`, `MLFloat16`)
- Other parameters same as `ONNX_OPERATOR_KERNEL_EX`

**Example:**
```cpp
ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    int8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<int8_t, int8_t>);
```

#### 4. `ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX`
Defines a type-specific kernel for a range of opset versions.

```cpp
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, domain, startver, endver, type, provider, builder, kernel_class)
```

### Additional Typed Macros

For operators requiring multiple type constraints:

- `ONNX_OPERATOR_TWO_TYPED_KERNEL_EX`: For kernels with two type parameters
- `ONNX_OPERATOR_THREE_TYPED_KERNEL_EX`: For kernels with three type parameters
- `ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX`: Versioned two-typed kernel

## Registration Steps

When adding a new kernel or extending an existing one with new types/versions, follow these three mandatory steps:

### Step 1: Forward Declaration in Execution Provider

Add a forward declaration in the execution provider file (e.g., `cuda_execution_provider.cc`, `cpu_execution_provider.cc`).

**Location:** Near the top of the file, after includes and before the registration function.

```cpp
// Example from cuda_execution_provider.cc
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger);
```

**Purpose:** This declares the existence of the kernel class without defining it, allowing it to be referenced in the registration table.

### Step 2: Add BuildKernelCreateInfo Entry

Add a `BuildKernelCreateInfo` call in the kernel registration function of the execution provider.

**Location:** Inside the `RegisterOnnxOperatorKernels()` function (or similar), in the `function_table` array.

```cpp
// Example from cuda_execution_provider.cc
Status RegisterOnnxOperatorKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      // ... other kernels ...
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger)>,
      // ... more kernels ...
  };
  
  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  
  return Status::OK();
}
```

**Purpose:** This entry tells the registration system to invoke the template specialization that will create the kernel metadata and factory function.

### Step 3: Kernel Implementation with Macro

Define the actual kernel implementation in its own file using the appropriate `ONNX_OPERATOR_*` macro.

**Location:** In the kernel's implementation file (e.g., `matmul_integer.cc`).

```cpp
// Example from matmul_integer.cc
ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    int8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<int8_t, int8_t>);
```

**Purpose:** This macro expands to:
1. A forward declaration of the kernel class
2. A template specialization of `BuildKernelCreateInfo` that returns a `KernelCreateInfo` object
3. The kernel metadata (name, domain, version, type constraints)
4. A factory function to create instances of the kernel

## Complete Example: MatMulInteger

This section shows a complete example of registering the `MatMulInteger` operator for the CUDA execution provider.

### File 1: cuda_execution_provider.cc

```cpp
// Forward declaration (near line 542)
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger);

// Registration entry (near line 1610)
Status RegisterOnnxOperatorKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      // ... other kernels ...
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(
          kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger)>,
      // ... more kernels ...
  };
  
  // Registration loop
  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }
  
  return Status::OK();
}
```

### File 2: matmul_integer.h

```cpp
#pragma once
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T1, typename T2>
class MatMulInteger final : public CudaKernel {
  using Base = CudaKernel;

 public:
  MatMulInteger(const OpKernelInfo& info) : CudaKernel(info) {
    has_a_zero_point_ = false;
    has_b_zero_point_ = false;
    if (info.GetInputCount() > 2) {
      has_a_zero_point_ = true;
    }
    if (info.GetInputCount() > 3) {
      has_b_zero_point_ = true;
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  bool has_a_zero_point_;
  bool has_b_zero_point_;
};

}  // namespace cuda
}  // namespace onnxruntime
```

### File 3: matmul_integer.cc

```cpp
#include "matmul_integer.h"
#include "matmul_integer.cuh"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/shared_inc/integer_gemm.h"

namespace onnxruntime {
namespace cuda {

// Kernel registration macro
ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    int8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    MatMulInteger<int8_t, int8_t>);

// Kernel implementation
template <>
Status MatMulInteger<int8_t, int8_t>::ComputeInternal(OpKernelContext* ctx) const {
  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* Y = ctx->Output(0, helper.OutputShape());

  if (Y->Shape().Size() == 0)
    return Status::OK();

  // Implementation details...
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
```

## OpSet Versioning

ONNX operators evolve over time through different opset versions. ONNX Runtime supports this through version-aware kernel registration.

### Single Version Registration

Use when a kernel supports exactly one opset version:

```cpp
// Supports only opset version 13
ONNX_OPERATOR_KERNEL_EX(
    Softmax,
    kOnnxDomain,
    13,
    kCudaExecutionProvider,
    builder,
    Softmax);
```

**Class name macro:**
```cpp
ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, Softmax)
```

**When to use:**
- The operator implementation is specific to one opset version
- Different opset versions require different implementations
- Latest version that doesn't need backward compatibility with older versions

### Version Range Registration

Use when a kernel supports multiple consecutive opset versions with the same implementation:

```cpp
// Supports opset versions 6 through 12 (inclusive)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Relu,
    kOnnxDomain,
    6,      // start version (inclusive)
    12,     // end version (inclusive)
    kCpuExecutionProvider,
    builder,
    Relu<float>);
```

**Class name macro:**
```cpp
ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, Relu)
```

**When to use:**
- The operator behavior is identical across multiple opset versions
- You want to avoid duplicating code for each version
- The operator spec hasn't changed in a meaningful way across versions

### Key Differences

| Aspect | Single Version | Version Range |
|--------|---------------|---------------|
| **Macro** | `ONNX_OPERATOR_KERNEL_EX` | `ONNX_OPERATOR_VERSIONED_KERNEL_EX` |
| **Class Name** | `..._ver13` | `..._ver6_12` |
| **Version Parameter** | Single `ver` | `startver` and `endver` |
| **Matching Behavior** | **Exact match ONLY** | Matches any version in range (inclusive) |
| **Future Compatibility** | **Does NOT match future versions** | Matches all versions up to `endver` |
| **Internal End Version** | `INT_MAX` (unbounded) | Specified `endver` value |
| **Use Case** | Version-specific behavior | Backward compatibility |

### Critical Understanding: Version Matching Behavior

⚠️ **Important:** The difference between single version and version range registration is **NOT** just syntactic sugar—it fundamentally changes which opset versions the kernel will match.

**Single Version Registration:**
```cpp
ONNX_OPERATOR_KERNEL_EX(..., 10, ...)  // ver = 10
```
- **Only matches opset version 10** (exact match)
- Will **NOT** match versions 11, 12, 13, etc.
- Internally sets `kernel_end_version = INT_MAX` but requires exact match
- Use when you've only tested/support one specific version

**Version Range Registration:**
```cpp
ONNX_OPERATOR_VERSIONED_KERNEL_EX(..., 10, 15, ...)  // startver = 10, endver = 15
```
- Matches opset versions 10, 11, 12, 13, 14, and 15 (inclusive range)
- Internally sets `kernel_start_version = 10` and `kernel_end_version = 15`
- Use when operator spec is stable across multiple versions

**Why This Matters:**

From the kernel registry's `VerifyVersion` function:
```cpp
bool valid_version =
    // Exact match case (for single version kernels)
    kernel_start_version == since_ver ||
    
    // Range match case (only if kernel has explicit end version)
    (kernel_end_version != INT_MAX &&
     kernel_start_version <= since_ver && 
     kernel_end_version >= since_ver);
```

The check `kernel_end_version != INT_MAX` means:
- Single version kernels (with implicit `INT_MAX` end version) **only match via exact match**
- Version range kernels (with explicit end version) can match any version in their range

This design is **intentional** and serves as a **safety mechanism**: ONNX Runtime doesn't assume your kernel supports future opset versions unless you explicitly declare the range.

### Handling Multiple Versions

When an operator changes behavior across versions, register separate kernels:

```cpp
// Old behavior for opset 6-12
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    Relu,
    kOnnxDomain,
    6,
    12,
    float,
    kCpuExecutionProvider,
    builder_v6_12,
    ReluOld<float>);

// New behavior for opset 13+
ONNX_OPERATOR_TYPED_KERNEL_EX(
    Relu,
    kOnnxDomain,
    13,
    float,
    kCpuExecutionProvider,
    builder_v13,
    ReluNew<float>);
```

**Forward declarations:**
```cpp
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Relu);
```

### Version Lookup

At runtime, ONNX Runtime uses specific logic to match kernels to operators based on version. The matching algorithm is implemented in the `VerifyVersion` function in `kernel_registry.cc`.

#### Important: Node Version vs Model Opset

⚠️ **Critical Understanding:** The version used for matching is the **operator's version** (when it was introduced/last changed in ONNX spec), **NOT** the model's opset version.

**Example Scenario:**
- MatMulInteger introduced in opset 10
- No changes in opset 11, 12
- Changed in opset 13

```cpp
// If you have an opset 11 model with MatMulInteger:
Node {
  op_type: "MatMulInteger"
  since_version: 10  // ← Still 10! (operator unchanged since opset 10)
}

// Even though model declares opset 11, the node's since_version is 10
```

**What this means for kernel registration:**
```cpp
// Single version registration for version 10
ONNX_OPERATOR_KERNEL_EX(MatMulInteger, kOnnxDomain, 10, ...)

// ✓ Works with opset 10 models (node since_version = 10)
// ✓ Works with opset 11 models (node since_version = 10, operator unchanged)
// ✓ Works with opset 12 models (node since_version = 10, operator unchanged)
// ✗ Fails with opset 13 models (node since_version = 13, operator changed!)
```

This is why single version registration often works across multiple model opsets - as long as the operator spec hasn't changed!

#### The Matching Algorithm

```cpp
bool valid_version =
    // exact match. typical usage.
    kernel_start_version == since_ver ||
    // allow match if the kernel def has an end version. if it does not, all we know is that the kernel supported
    // the start version when it was created, and not whether a new version of the operator was added since then
    // that the kernel doesn't support.
    (kernel_end_version != INT_MAX &&
     kernel_start_version <= since_ver && kernel_end_version >= since_ver);
```

**Key matching rules:**

1. **Exact match is always valid**: If `kernel_start_version == node_version`, the kernel is used (typical case)

2. **Range match requires explicit end version**: A kernel registered with a version range will only match if:
   - The kernel has an explicit end version (`kernel_end_version != INT_MAX`)
   - The node version falls within the range: `kernel_start_version <= node_version <= kernel_end_version`

3. **Single version registration is conservative**: If you register a kernel with only a start version (no end version), it will **ONLY** match that exact version. This is a safety feature because:
   - The kernel was only tested with that specific opset version
   - Future opset versions of the operator may introduce breaking changes
   - ONNX Runtime cannot assume the kernel supports newer versions without explicit declaration

**Important Implication:**

```cpp
// This kernel ONLY matches opset version 10 (exact match only)
ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,  // Only matches version 10, not 11, 12, etc.
    int8_t,
    kCudaExecutionProvider,
    builder,
    MatMulInteger<int8_t, int8_t>);

// This kernel matches opset versions 6, 7, 8, 9, 10, 11, and 12
ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    Relu,
    kOnnxDomain,
    6,    // start version
    12,   // end version (explicitly set, not INT_MAX)
    float,
    kCpuExecutionProvider,
    builder,
    Relu<float>);
```

This design ensures backward compatibility is explicitly declared rather than assumed.

## Common Pitfalls and Solutions

### Pitfall 1: Expecting Single Version to Match Future Versions

**Incorrect Assumption:**
```cpp
// Registered kernel for version 10
ONNX_OPERATOR_KERNEL_EX(MyOp, kOnnxDomain, 10, kCudaExecutionProvider, builder, MyOp);

// Developer expects this to work with a model using opset 11, 12, etc.
// ⚠️ DEPENDS on whether the operator changed in those opsets!
```

**What Actually Happens:**

If the operator is **unchanged** in newer opsets:
```cpp
// Model opset 11, but MyOp unchanged since opset 10
// Node in model will have since_version = 10
// ✓ Kernel matches! (10 == 10)
```

If the operator **changed** in newer opsets:
```cpp
// Model opset 13, and MyOp changed in opset 13
// Node in model will have since_version = 13
// ✗ Kernel fails to match! (10 ≠ 13)
```

**Best Practice Solution:**

When you know the operator won't change for a range of opsets, be explicit:
```cpp
// Supports all opsets where operator is unchanged (10-12)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 10, 12, kCudaExecutionProvider, builder, MyOp);

// If operator changes in opset 13, add new registration:
ONNX_OPERATOR_KERNEL_EX(MyOp, kOnnxDomain, 13, kCudaExecutionProvider, builder, MyOpV2);
```

This makes your intent clear and prevents issues when the ONNX spec evolves.

### Pitfall 2: Gaps in Version Coverage

**Problem:**
```cpp
// Supports version 1-5
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 1, 5, ...);

// Supports version 10 only
ONNX_OPERATOR_KERNEL_EX(MyOp, kOnnxDomain, 10, ...);

// ❌ Models with opset versions 6-9 will FAIL - no kernel available!
```

**Solution:**
Ensure continuous coverage:
```cpp
// Version 1-5
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 1, 5, ..., MyOpV1);

// Version 6-9 (bridge the gap)
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 6, 9, ..., MyOpV6);

// Version 10+
ONNX_OPERATOR_KERNEL_EX(MyOp, kOnnxDomain, 10, ..., MyOpV10);
```

### Pitfall 3: Overlapping Version Ranges

**Problem:**
```cpp
// Both kernels claim to support version 10
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 1, 10, ...);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 10, 15, ...);

// ❌ Registration will fail with conflict error!
```

**Solution:**
Make ranges non-overlapping:
```cpp
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 1, 9, ...);   // Up to 9
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 10, 15, ...); // From 10
```

### Pitfall 4: Not Updating Range When Opset Changes

**Problem:**
```cpp
// Initially registered for versions 1-12
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 1, 12, ..., MyOp);

// New ONNX opset 13 introduces breaking change to MyOp
// ❌ Old kernel still registered up to version 12, but should be updated
```

**Solution:**
When opset changes operator behavior, close the old range and add new kernel:
```cpp
// Old implementation for versions 1-12
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 1, 12, ..., MyOpOld);

// New implementation for version 13+
ONNX_OPERATOR_KERNEL_EX(MyOp, kOnnxDomain, 13, ..., MyOpNew);
```

## Best Practices

### 1. Use Type-Specific Macros for Type Specialization

When supporting multiple data types, use typed macros:

```cpp
// Good: Separate registration for each type
ONNX_OPERATOR_TYPED_KERNEL_EX(..., float, ..., Relu<float>);
ONNX_OPERATOR_TYPED_KERNEL_EX(..., double, ..., Relu<double>);
ONNX_OPERATOR_TYPED_KERNEL_EX(..., MLFloat16, ..., Relu<MLFloat16>);

// Avoid: Generic registration (loses type information)
```

### 2. Keep Forward Declarations Consistent

Ensure the forward declaration exactly matches the registration macro:

```cpp
// Forward declaration
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger);

// Must match in BuildKernelCreateInfo
BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 10, int8_t, MatMulInteger)>

// Must match in implementation
ONNX_OPERATOR_TYPED_KERNEL_EX(MatMulInteger, kOnnxDomain, 10, int8_t, kCudaExecutionProvider, ...)
```

### 3. Use KernelDefBuilder for Constraints

Properly specify type constraints and memory requirements:

```cpp
(*KernelDefBuilder::Create())
    .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
    .InputMemoryType(OrtMemTypeCPUInput, 0)  // Input 0 must be in CPU memory
    .OutputMemoryType(OrtMemTypeCPUOutput, 0)  // Output 0 goes to CPU memory
```

### 4. Organize Forward Declarations

Group related forward declarations together:

```cpp
// Relu variants
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, float, Relu);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 6, 12, double, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCpuExecutionProvider, kOnnxDomain, 13, double, Relu);
```

### 5. Handle Optional Inputs

When operators have optional inputs, check their existence in the kernel:

```cpp
MatMulInteger(const OpKernelInfo& info) : CudaKernel(info) {
  has_a_zero_point_ = (info.GetInputCount() > 2);
  has_b_zero_point_ = (info.GetInputCount() > 3);
}
```

### 6. Version Range Best Practices

- **End version inclusive**: When using version ranges, the end version is inclusive
- **No gaps**: Ensure version ranges don't leave gaps in coverage
- **Document changes**: Comment why version ranges change

```cpp
// Supports opset 1-10 with old implementation
ONNX_OPERATOR_VERSIONED_KERNEL_EX(..., 1, 10, ...)

// Supports opset 11+ with new implementation (new attribute added in v11)
ONNX_OPERATOR_KERNEL_EX(..., 11, ...)
```

### 7. Conditional Registration

Use `#ifdef` for platform-specific kernels:

```cpp
#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(
    kCpuExecutionProvider, kOnnxDomain, 6, 12, MLFloat16, Relu);
#endif
```

### 8. Kernel Lookup Efficiency

The kernel registry uses hash maps for efficient lookup based on:
- Operator name
- Domain
- Opset version  
- Execution provider
- Type constraints

Proper registration ensures O(1) lookup performance.

## Common Domains

All operator domains are defined in `include/onnxruntime/core/graph/constants.h`.

| Domain Constant | Value | Description |
|----------------|-------|-------------|
| `kOnnxDomain` | `""` (empty string) | Standard ONNX operators (default domain) |
| `kOnnxDomainAlias` | `"ai.onnx"` | Alias for ONNX domain (converted to empty string internally) |
| `kMLDomain` | `"ai.onnx.ml"` | ONNX ML operators (classical machine learning) |
| `kMSDomain` | `"com.microsoft"` | Microsoft-specific operators |
| `kMSExperimentalDomain` | `"com.microsoft.experimental"` | Microsoft experimental operators |
| `kMSNchwcDomain` | `"com.microsoft.nchwc"` | Microsoft NCHWC (channels-first) operators |
| `kMSInternalNHWCDomain` | `"com.ms.internal.nhwc"` | Microsoft internal NHWC (channels-last) operators |
| `kMSDmlDomain` | `"com.microsoft.dml"` | Microsoft DirectML operators |
| `kPytorchAtenDomain` | `"org.pytorch.aten"` | PyTorch ATen operators |
| `kNGraphDomain` | `"com.intel.ai"` | Intel nGraph operators |
| `kVitisAIDomain` | `"com.xilinx"` | Xilinx Vitis AI operators |
| `kMIGraphXDomain` | `""` (empty string) | AMD MIGraphX operators |

**Note:** `kOnnxDomainAlias` ("ai.onnx") is automatically converted to `kOnnxDomain` (empty string) by `Node::Init()`, so all Node instances internally use the empty string for the ONNX domain.

**Usage in kernel registration:**
```cpp
// Standard ONNX operator
ONNX_OPERATOR_KERNEL_EX(Relu, kOnnxDomain, 13, kCudaExecutionProvider, ...)

// Microsoft domain operator
ONNX_OPERATOR_KERNEL_EX(FusedConv, kMSDomain, 1, kCudaExecutionProvider, ...)

// ML domain operator  
ONNX_OPERATOR_KERNEL_EX(TreeEnsembleClassifier, kMLDomain, 1, kCpuExecutionProvider, ...)
```

## Common Execution Providers

All execution provider names are defined in `include/onnxruntime/core/graph/constants.h`.

| Provider Constant | Value | Description |
|------------------|-------|-------------|
| `kCpuExecutionProvider` | `"CPUExecutionProvider"` | CPU execution (default, always available) |
| `kCudaExecutionProvider` | `"CUDAExecutionProvider"` | NVIDIA CUDA (GPU) |
| `kCudaNHWCExecutionProvider` | `"CUDANHWCExecutionProvider"` | CUDA with NHWC layout optimization |
| `kRocmExecutionProvider` | `"ROCMExecutionProvider"` | AMD ROCm (GPU) |
| `kDmlExecutionProvider` | `"DmlExecutionProvider"` | DirectML (Windows GPU acceleration) |
| `kTensorrtExecutionProvider` | `"TensorrtExecutionProvider"` | NVIDIA TensorRT |
| `kNvTensorRTRTXExecutionProvider` | `"NvTensorRTRTXExecutionProvider"` | NVIDIA TensorRT RTX |
| `kOpenVINOExecutionProvider` | `"OpenVINOExecutionProvider"` | Intel OpenVINO |
| `kDnnlExecutionProvider` | `"DnnlExecutionProvider"` | Intel oneDNN (formerly DNNL/MKL-DNN) |
| `kMIGraphXExecutionProvider` | `"MIGraphXExecutionProvider"` | AMD MIGraphX |
| `kNnapiExecutionProvider` | `"NnapiExecutionProvider"` | Android Neural Networks API |
| `kQnnExecutionProvider` | `"QNNExecutionProvider"` | Qualcomm AI Engine Direct |
| `kSnpeExecutionProvider` | `"SNPEExecutionProvider"` | Qualcomm Snapdragon NPE |
| `kCoreMLExecutionProvider` | `"CoreMLExecutionProvider"` | Apple Core ML |
| `kAclExecutionProvider` | `"ACLExecutionProvider"` | Arm Compute Library |
| `kArmNNExecutionProvider` | `"ArmNNExecutionProvider"` | Arm NN |
| `kVitisAIExecutionProvider` | `"VitisAIExecutionProvider"` | Xilinx Vitis AI |
| `kRknpuExecutionProvider` | `"RknpuExecutionProvider"` | Rockchip NPU |
| `kVSINPUExecutionProvider` | `"VSINPUExecutionProvider"` | Verisilicon NPU |
| `kCannExecutionProvider` | `"CANNExecutionProvider"` | Huawei CANN (Ascend) |
| `kTvmExecutionProvider` | `"TvmExecutionProvider"` | Apache TVM |
| `kXnnpackExecutionProvider` | `"XnnpackExecutionProvider"` | XNNPACK (optimized for mobile/web) |
| `kJsExecutionProvider` | `"JsExecutionProvider"` | JavaScript/WebAssembly |
| `kWebNNExecutionProvider` | `"WebNNExecutionProvider"` | Web Neural Network API |
| `kWebGpuExecutionProvider` | `"WebGpuExecutionProvider"` | WebGPU |
| `kAzureExecutionProvider` | `"AzureExecutionProvider"` | Azure-specific optimizations |

**Most commonly used providers:**
- **CPU**: Universal, always available
- **CUDA**: NVIDIA GPUs (most common GPU backend)
- **ROCm**: AMD GPUs
- **TensorRT**: NVIDIA optimized inference
- **OpenVINO**: Intel hardware (CPU, GPU, VPU, FPGA)
- **DirectML**: Windows GPU acceleration (vendor-agnostic)
- **CoreML**: Apple devices (iOS, macOS)

**Usage in kernel registration:**
```cpp
// CPU kernel
ONNX_OPERATOR_KERNEL_EX(Relu, kOnnxDomain, 13, kCpuExecutionProvider, ...)

// CUDA kernel  
ONNX_OPERATOR_KERNEL_EX(MatMul, kOnnxDomain, 13, kCudaExecutionProvider, ...)

// TensorRT kernel
ONNX_OPERATOR_KERNEL_EX(Conv, kOnnxDomain, 11, kTensorrtExecutionProvider, ...)
```

**Note:** Maximum execution provider name length is 30 characters (defined by `kMaxExecutionProviderNameLen`).

## Debugging Registration Issues

### Common Errors

1. **Mismatch between forward declaration and registration**
   - Error: Unresolved symbol during compilation
   - Fix: Ensure exact match of all parameters in all three locations

2. **Missing BuildKernelCreateInfo entry**
   - Error: Kernel not found at runtime
   - Fix: Add entry to function_table in execution provider

3. **Type constraint mismatch**
   - Error: No kernel found for type
   - Fix: Check TypeConstraint in KernelDefBuilder matches actual types

4. **Version not covered**
   - Error: `No kernel for opset version X`
   - Common cause: Using single version registration when you meant to support a range
   - Fix: Use `VERSIONED` macro or add additional registrations to cover the gap

5. **Assuming single version matches future versions**
   - Error: `Version mismatch. node_version: 12 kernel start version: 10 kernel_end_version: 2147483647`
   - Cause: Registered with `ONNX_OPERATOR_KERNEL_EX(..., 10, ...)` but model uses version 12
   - Fix: Use `ONNX_OPERATOR_VERSIONED_KERNEL_EX(..., 10, 15, ...)` to explicitly support range

6. **Conflicting kernel versions**
   - Error: `Failed to add kernel for X: Conflicting with a registered kernel with op versions`
   - Cause: Two kernels registered with overlapping version ranges
   - Fix: Ensure version ranges don't overlap

### Understanding Error Messages

**"Version mismatch" error details:**
```
Version mismatch.
node_version: 12
kernel start version: 10
kernel_end_version: 2147483647
```

What this means:
- `node_version: 12` - Your model uses opset version 12
- `kernel start version: 10` - Kernel was registered for version 10
- `kernel_end_version: 2147483647` - This is `INT_MAX`, meaning single version registration
- **Problem**: Single version kernels only match exact version, so version 12 ≠ version 10

**Solution:** Change from:
```cpp
ONNX_OPERATOR_KERNEL_EX(MyOp, kOnnxDomain, 10, ...)  // Only version 10
```
To:
```cpp
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 10, 15, ...)  // Versions 10-15
```

### Verification Steps

1. Check all three registration points are present
2. Verify forward declaration matches exactly
3. Confirm BuildKernelCreateInfo is in function_table
4. Validate macro parameters (especially types and versions)
5. Test with a model using the operator

## Conclusion

Kernel registration in ONNX Runtime requires careful coordination between:
- Forward declarations in the execution provider file
- Registration entries in the kernel registry
- Implementation with appropriate macros

By following the three-step process and using the correct macros for your use case, you can successfully register custom kernels or extend existing ones with new types and versions.

For more examples, search for existing operators in the codebase:
- CPU kernels: `onnxruntime/core/providers/cpu/`
- CUDA kernels: `onnxruntime/core/providers/cuda/`
- Other providers: `onnxruntime/core/providers/<provider_name>/`

---

## Quick Reference: Version Matching Rules

### The Golden Rule

```
Single Version Registration = EXACT MATCH ONLY
Versioned Registration = RANGE MATCH
```

### Visual Guide

```
Model opset version:     1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
                         │   │   │   │   │   │   │   │   │   │   │   │   │   │   │

Single Version (ver=10): ─ ─ ─ ─ ─ ─ ─ ─ ─ ✓ ─ ─ ─ ─ ─
Only matches version 10                     ↑
                                            exact match only

Versioned (6,12):        ─ ─ ─ ─ ─ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ─ ─ ─
Matches versions 6-12              ↑───────────────↑
                                   inclusive range

Versioned (10,15):       ─ ─ ─ ─ ─ ─ ─ ─ ─ ✓ ✓ ✓ ✓ ✓ ✓
Matches versions 10-15                     ↑───────────────↑
                                           inclusive range
```

### Decision Tree

```
Do you want to support multiple opset versions?
│
├─ YES → Use ONNX_OPERATOR_VERSIONED_KERNEL_EX
│        with explicit start and end versions
│        Example: VERSIONED_KERNEL_EX(..., 10, 15, ...)
│
└─ NO  → Use ONNX_OPERATOR_KERNEL_EX
         with single version (exact match only)
         Example: KERNEL_EX(..., 10, ...)
```

### Cheat Sheet

| What I Want | What To Use | Example |
|-------------|-------------|---------|
| Support ONLY version 10 | `KERNEL_EX` | `ONNX_OPERATOR_KERNEL_EX(Op, kOnnxDomain, 10, ...)` |
| Support versions 10-15 | `VERSIONED_KERNEL_EX` | `ONNX_OPERATOR_VERSIONED_KERNEL_EX(Op, kOnnxDomain, 10, 15, ...)` |
| Support version 10 onwards (open-ended) | Multiple registrations | Register highest known version with VERSIONED |
| Update for new opset | Close old range, add new | See "Handling Multiple Versions" section |

### Code Snippet: VerifyVersion Logic

From `kernel_registry.cc` - this is what actually runs at runtime:

```cpp
static bool VerifyVersion(int since_ver, const KernelDef& kernel_def, std::string& error_str) {
  int kernel_start_version;
  int kernel_end_version;
  kernel_def.SinceVersion(&kernel_start_version, &kernel_end_version);

  bool valid_version =
      // exact match. typical usage.
      kernel_start_version == since_ver ||
      
      // allow match if the kernel def has an end version. if it does not, all we know is 
      // that the kernel supported the start version when it was created, and not whether 
      // a new version of the operator was added since then that the kernel doesn't support.
      (kernel_end_version != INT_MAX &&
       kernel_start_version <= since_ver && 
       kernel_end_version >= since_ver);

  return valid_version;
}
```

**Key takeaway:** The `kernel_end_version != INT_MAX` check is what makes single-version registration only match exact versions!

## Frequently Asked Questions (FAQ)

### Q1: What's the difference between "model opset version" and "node version"?

**Answer:** They are different concepts:

- **Model opset version**: Declared at the model level (e.g., "This model uses ONNX opset 11")
- **Node version (since_version)**: The version when that specific operator was introduced or last changed in the ONNX spec

**Example:**
```
Model: opset 11
├─ Relu node: since_version = 6 (last changed in opset 6)
├─ MatMul node: since_version = 1 (unchanged since opset 1)  
└─ MatMulInteger node: since_version = 10 (introduced in opset 10)
```

The kernel registry matches against **node version**, not model opset!

### Q2: I registered my kernel for opset 10, will it work with opset 11 models?

**Answer:** It depends on whether the operator changed between opset 10 and 11.

**Case 1: Operator unchanged**
```cpp
// Operator introduced in opset 10, unchanged in 11
// Model opset 11 will have nodes with since_version = 10
ONNX_OPERATOR_KERNEL_EX(MyOp, kOnnxDomain, 10, ...)  
// ✓ Works! Node version is 10, kernel is 10
```

**Case 2: Operator changed**
```cpp
// Operator changed in opset 11
// Model opset 11 will have nodes with since_version = 11
ONNX_OPERATOR_KERNEL_EX(MyOp, kOnnxDomain, 10, ...)  
// ✗ Fails! Node version is 11, kernel is 10
```

**Best practice:** Use versioned registration to be explicit:
```cpp
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 10, 12, ...)
```

### Q3: Why does single version registration have `kernel_end_version = INT_MAX`?

**Answer:** This is for internal representation, but the `kernel_end_version != INT_MAX` check prevents it from matching as a range.

```cpp
// Single version sets end_version = INT_MAX internally
// BUT the VerifyVersion check requires:
(kernel_end_version != INT_MAX && ...)  // This is false for single version!

// So single version can ONLY match via exact match:
kernel_start_version == since_ver
```

This design ensures you explicitly declare version ranges rather than accidentally supporting untested versions.

### Q4: When should I use VERSIONED vs regular registration?

**Decision tree:**

```
Has the operator spec changed across opsets?
│
├─ NO (operator stable across opsets X-Y)
│  └─ Use: ONNX_OPERATOR_VERSIONED_KERNEL_EX(..., X, Y, ...)
│     Benefit: Explicitly documents supported range
│
├─ YES (operator changed at specific opset)
│  └─ Use separate registrations:
│     - ONNX_OPERATOR_VERSIONED_KERNEL_EX(..., old_start, old_end, ..., OldImpl)
│     - ONNX_OPERATOR_KERNEL_EX(..., new_version, ..., NewImpl)
│
└─ UNSURE (just adding new operator)
   └─ Use: ONNX_OPERATOR_KERNEL_EX(..., current_opset, ...)
      Then update when spec evolves
```

### Q5: What happens if I have gaps in version coverage?

**Answer:** Models with operators in the gap will fail to find a kernel.

```cpp
// Registered: opset 1-5
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 1, 5, ...)

// Registered: opset 10
ONNX_OPERATOR_KERNEL_EX(MyOp, kOnnxDomain, 10, ...)

// Gap: opset 6-9 have NO kernel!
// Models with opset 6-9 containing MyOp will fail: "Kernel not found"
```

**Solution:** Ensure continuous coverage by bridging gaps:
```cpp
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 1, 5, ..., MyOpV1);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(MyOp, kOnnxDomain, 6, 9, ..., MyOpV6);
ONNX_OPERATOR_KERNEL_EX(MyOp, kOnnxDomain, 10, ..., MyOpV10);
```

### Q6: Can I see what version a node has in my model?

**Answer:** Yes, you can inspect the ONNX model:

```python
import onnx

model = onnx.load("model.onnx")
for node in model.graph.node:
    print(f"{node.op_type}: domain={node.domain or 'onnx'}, version={model.opset_import[0].version}")
    
# Or use tools like Netron to visualize
```

Note: The node doesn't store its own version; it inherits from the opset import based on when the operator was defined in that opset.

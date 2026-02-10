// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/telum/telum_kernel_registry.h"

#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/graph/constants.h"

namespace onnxruntime {
namespace telum {
namespace {

#define KERNEL_CREATE_INFO(Ver, Op) \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, Ver, Op)>

#define KERNEL_CREATE_INFO_VERSIONED(Start, End, Op) \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, Start, End, Op)>

#define KERNEL_CREATE_INFO_MS(Ver, Op) \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kMSDomain, Ver, Op)>

// Kernel class forward declarations. These are the macro-generated "wrapper" kernel class names
// (not the implementation class name), so we can refer to them in BuildKernelCreateInfo<>.
//
// MatMul
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 1, 12, MatMul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, MatMul);

// Gemm
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 7, 8, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 9, 10, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 11, 12, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, Gemm);

// Elementwise
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 7, 12, Add);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, 13, Add);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 14, Add);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 7, 12, Sub);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, 13, Sub);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 14, Sub);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 7, 12, Mul);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, 13, Mul);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 14, Mul);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 7, 12, Div);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, 13, Div);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 14, Div);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 8, 11, Min);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 12, 12, Min);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, Min);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 8, 11, Max);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 12, 12, Max);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, Max);

// Activations
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 6, 12, Relu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, 13, Relu);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 14, Relu);

// Softmax
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, Softmax);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kMSDomain, 1, Gelu);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 6, 12, Tanh);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, Tanh);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 6, 12, Sigmoid);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, Sigmoid);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 6, 12, Exp);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, Exp);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 6, 12, Log);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, Log);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 6, 12, Sqrt);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 13, Sqrt);

// Normalization
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTelumExecutionProvider, kOnnxDomain, 17, LayerNormalization);

void RegisterTelumKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      // MatMul
      KERNEL_CREATE_INFO_VERSIONED(1, 12, MatMul),
      KERNEL_CREATE_INFO(13, MatMul),

      // Gemm
      KERNEL_CREATE_INFO_VERSIONED(7, 8, Gemm),
      KERNEL_CREATE_INFO_VERSIONED(9, 10, Gemm),
      KERNEL_CREATE_INFO_VERSIONED(11, 12, Gemm),
      KERNEL_CREATE_INFO(13, Gemm),

      // Elementwise
      KERNEL_CREATE_INFO_VERSIONED(7, 12, Add),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Add),
      KERNEL_CREATE_INFO(14, Add),

      KERNEL_CREATE_INFO_VERSIONED(7, 12, Sub),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Sub),
      KERNEL_CREATE_INFO(14, Sub),

      KERNEL_CREATE_INFO_VERSIONED(7, 12, Mul),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Mul),
      KERNEL_CREATE_INFO(14, Mul),

      KERNEL_CREATE_INFO_VERSIONED(7, 12, Div),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Div),
      KERNEL_CREATE_INFO(14, Div),

      KERNEL_CREATE_INFO_VERSIONED(8, 11, Min),
      KERNEL_CREATE_INFO_VERSIONED(12, 12, Min),
      KERNEL_CREATE_INFO(13, Min),

      KERNEL_CREATE_INFO_VERSIONED(8, 11, Max),
      KERNEL_CREATE_INFO_VERSIONED(12, 12, Max),
      KERNEL_CREATE_INFO(13, Max),

      // Activations
      KERNEL_CREATE_INFO_VERSIONED(6, 12, Relu),
      KERNEL_CREATE_INFO_VERSIONED(13, 13, Relu),
      KERNEL_CREATE_INFO(14, Relu),

      // Softmax
      KERNEL_CREATE_INFO(13, Softmax),

      KERNEL_CREATE_INFO_MS(1, Gelu),

      KERNEL_CREATE_INFO_VERSIONED(6, 12, Tanh),
      KERNEL_CREATE_INFO(13, Tanh),

      KERNEL_CREATE_INFO_VERSIONED(6, 12, Sigmoid),
      KERNEL_CREATE_INFO(13, Sigmoid),

      KERNEL_CREATE_INFO_VERSIONED(6, 12, Exp),
      KERNEL_CREATE_INFO(13, Exp),

      KERNEL_CREATE_INFO_VERSIONED(6, 12, Log),
      KERNEL_CREATE_INFO(13, Log),

      KERNEL_CREATE_INFO_VERSIONED(6, 12, Sqrt),
      KERNEL_CREATE_INFO(13, Sqrt),

      // Normalization
      KERNEL_CREATE_INFO(17, LayerNormalization),
  };

  for (const auto& fn : function_table) {
    ORT_ENFORCE(kernel_registry.Register(fn()).IsOK());
  }
}

}  // namespace

std::shared_ptr<KernelRegistry> GetTelumKernelRegistry() {
  static std::shared_ptr<KernelRegistry> kernel_registry = []() {
    auto registry = std::make_shared<KernelRegistry>();
    RegisterTelumKernels(*registry);
    return registry;
  }();

  return kernel_registry;
}

}  // namespace telum
}  // namespace onnxruntime

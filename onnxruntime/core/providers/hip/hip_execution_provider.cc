// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hip_common.h"
#include "hip_execution_provider.h"
#include "hip_fence.h"
#include "hip_allocator.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/framework/memcpy.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/graph_utils.h"
#include "core/providers/hip/gpu_data_transfer.h"

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/hip/hip_contrib_kernels.h"
#endif

#ifdef ENABLE_TRAINING
#include "orttraining/training_ops/hip/hip_training_kernels.h"
#endif

using namespace onnxruntime::common;

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  Status st;
};
}  // namespace

namespace onnxruntime {
namespace hip {

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kHipExecutionProvider,
    KernelDefBuilder()
        .InputMemoryType<OrtMemTypeCPUInput>(0)
        .ExecQueueId(kHipStreamCopyIn)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kHipExecutionProvider,
    KernelDefBuilder()
        .OutputMemoryType<OrtMemTypeCPUOutput>(0)
        .ExecQueueId(kHipStreamCopyOut)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

// opset 1 to 9
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 4, 10, Concat);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 8, Flatten);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, Squeeze);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, Identity);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, Dropout);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, Gather);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, float, Gemm);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, double, Gemm);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, float, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, double, Gemm);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Gemm);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 8, float, MatMul);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 8, double, MatMul);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, MatMul);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Tile);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Tile);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Elu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, HardSigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, LeakyRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Relu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Selu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Sigmoid);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, float, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, double, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MLFloat16, Softsign);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Tanh);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, float, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, double, Softplus);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MLFloat16, Softplus);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, Softmax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Softmax);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, Pow);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, PRelu);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, bool, And);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, bool, Or);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, bool, Xor);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, int32_t, Sum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, int64_t, Sum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, uint32_t, Sum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, uint64_t, Sum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, float, Sum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, double, Sum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, int32_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, int64_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, uint32_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, uint64_t, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, float, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, double, Sum);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, MLFloat16, Sum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, float, Max);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, double, Max);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, float, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, double, Max);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, MLFloat16, Max);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, float, Min);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, double, Min);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, float, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, double, Min);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, MLFloat16, Min);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, float, Greater);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, double, Greater);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Greater);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 10, bool, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 10, int32_t, Equal);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 10, int64_t, Equal);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, Expand);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint32_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint64_t, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Greater);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint32_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint64_t, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, Add);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint32_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint64_t, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, Sub);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint32_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint64_t, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, Mul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint32_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint64_t, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, Div);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, uint8_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, uint16_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, uint32_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, uint64_t, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Abs);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int8_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int16_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int32_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int64_t, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Neg);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Floor);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Ceil);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Ceil);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 10, float, Clip);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Reciprocal);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Sqrt);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Log);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Exp);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Erf);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, bool, Not);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, float, BatchNormalization);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, double, BatchNormalization);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, BatchNormalization);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, BatchNormalization);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, BatchNormalization);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, BatchNormalization);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, float, LRN);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, double, LRN);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MLFloat16, LRN);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, Conv);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, Conv);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Conv);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ConvTranspose);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ConvTranspose);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ConvTranspose);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, double, AveragePool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, AveragePool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, double, GlobalAveragePool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalAveragePool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, 9, double, MaxPool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, 9, MLFloat16, MaxPool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 7, double, MaxPool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, double, GlobalMaxPool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalMaxPool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ArgMax);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMax);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ArgMin);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ArgMin);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMin);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceL1);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceL1);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceL1);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceL1);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceL2);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceL2);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceL2);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceL2);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMax);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMax);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMax);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMax);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMean);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMean);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMean);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMin);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMin);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMin);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMin);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceProd);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceProd);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceProd);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceProd);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceSum);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceSum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceSum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceLogSum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceLogSum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceLogSum);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceSumSquare);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceSumSquare);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceSumSquare);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceLogSumExp);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceLogSumExp);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceLogSumExp);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, float, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, double, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, MLFloat16, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, int8_t, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, int16_t, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, int32_t, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, int64_t, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, uint8_t, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, uint16_t, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, uint32_t, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, uint64_t, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, bool, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint8_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint16_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint32_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint64_t, Cast);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, bool, Cast);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 2, 10, float, Pad);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 2, 10, double, Pad);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 2, 10, MLFloat16, Pad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 5, Reshape);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 4, Reshape_1);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, Shape);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Tile);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Tile);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, Transpose);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, InstanceNormalization);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, InstanceNormalization);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, InstanceNormalization);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, RNN);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, RNN);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, RNN);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, GRU);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, GRU);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, GRU);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, LSTM);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, LSTM);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, LSTM);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 9, int32_t, Slice);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 9, int64_t, Slice);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 9, float, Slice);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, Compress);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, Flatten);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, float, Upsample);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, double, Upsample);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, Upsample);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, int32_t, Upsample);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, uint8_t, Upsample);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 2, 10, Split);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, ConstantOfShape);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int8_t, Shrink);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int16_t, Shrink);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, Shrink);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, Shrink);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint8_t, Shrink);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint16_t, Shrink);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint32_t, Shrink);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint64_t, Shrink);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Shrink);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, Shrink);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Shrink);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, float, Less);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, double, Less);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint32_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint64_t, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, Less);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Less);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, EyeLike);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, Scatter);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Where);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Where);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, Where);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, Where);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint8_t, Where);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, bool, NonZero);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint8_t, NonZero);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, NonZero);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, NonZero);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, NonZero);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 9, TopK);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, If);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, 8, Scan);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, Scan);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, Loop);

// opset 10
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, double, AveragePool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, AveragePool);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, Dropout);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, double, MaxPool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, MaxPool);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, NonMaxSuppression);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, float, Resize);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, double, Resize);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, Resize);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, int32_t, Resize);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, uint8_t, Resize);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, ReverseSequence);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, float, RoiAlign);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, double, RoiAlign);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, int32_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, int64_t, Slice);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, float, Slice);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, float, ThresholdedRelu);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, double, ThresholdedRelu);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, MLFloat16, ThresholdedRelu);
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, TopK);

// opset 11
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ArgMax);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ArgMax);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ArgMax);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ArgMin);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ArgMin);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ArgMin);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Compress);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Concat);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Flatten);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Gather);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, GatherElements);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Gemm);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Gemm);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Gemm);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, If);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Loop);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, NonMaxSuppression);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Range);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceL1);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceL1);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceL1);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceL1);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceL2);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceL2);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceL2);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceL2);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceLogSum);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceLogSum);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceLogSum);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceLogSumExp);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceLogSumExp);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceLogSumExp);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceMax);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceMax);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceMax);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceMax);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceMean);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceMean);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceMean);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceMean);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceMin);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceMin);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceMin);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceMin);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceProd);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceProd);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceProd);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceProd);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceSum);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceSum);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceSum);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceSum);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceSumSquare);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceSumSquare);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceSumSquare);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Scan);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, ScatterElements);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, Slice);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int64_t, Slice);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Slice);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Softmax);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Softmax);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Softmax);
// // class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Split);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Squeeze);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, TopK);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Unsqueeze);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Conv);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Conv);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Conv);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ConvTranspose);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ConvTranspose);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ConvTranspose);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, AveragePool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, AveragePool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, AveragePool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, MaxPool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, MaxPool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, MaxPool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Resize);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Resize);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Resize);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, Resize);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, uint8_t, Resize);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Clip);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Pad);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Pad);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Pad);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, bool, Equal);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, Equal);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int64_t, Equal);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Round);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Round);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Round);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, CumSum);

static Status RegisterHipKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, Unsqueeze)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 8, Flatten)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, Squeeze)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, Identity)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, Dropout)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, Gather)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, float, Gemm)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, double, Gemm)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, float, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, double, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, MLFloat16, Gemm)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 8, float, MatMul)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 8, double, MatMul)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 8, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 10, float, Clip)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Tile)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Tile)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Elu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, HardSigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, LeakyRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Selu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Sigmoid)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, float, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, double, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MLFloat16, Softsign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Tanh)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, float, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, double, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MLFloat16, Softplus)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Softmax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, Pow)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, PRelu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, bool, And)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, bool, Or)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, bool, Xor)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, int32_t, Sum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, int64_t, Sum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, uint32_t, Sum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, uint64_t, Sum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, float, Sum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, double, Sum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, int32_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, int64_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, uint32_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, uint64_t, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, float, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, double, Sum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, MLFloat16, Sum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, float, Max)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, double, Max)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, float, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, double, Max)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, MLFloat16, Max)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, float, Min)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, double, Min)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 7, MLFloat16, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, float, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, double, Min)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, MLFloat16, Min)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, float, Greater)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, double, Greater)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 10, bool, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 10, int32_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 10, int64_t, Equal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, Expand)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint32_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint64_t, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Greater)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint32_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint64_t, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, Add)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint32_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint64_t, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, Sub)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint32_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint64_t, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, Mul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, int64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint32_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, uint64_t, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, Div)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, uint8_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, uint16_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, uint32_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, uint64_t, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Abs)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int8_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int16_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int32_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, int64_t, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Neg)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Floor)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Ceil)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Reciprocal)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Sqrt)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Log)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Exp)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Erf)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, bool, Not)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, float, BatchNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, double, BatchNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, BatchNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, BatchNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, BatchNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, BatchNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, float, LRN)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, double, LRN)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MLFloat16, LRN)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, Conv)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, Conv)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, Conv)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ConvTranspose)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ConvTranspose)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ConvTranspose)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, double, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, double, GlobalAveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalAveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, 9, float, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, 9, double, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, 9, MLFloat16, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 7, double, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 7, MLFloat16, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, double, GlobalMaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, MLFloat16, GlobalMaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ArgMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ArgMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ArgMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ArgMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ArgMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceL1)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceL1)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceL1)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceL1)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceL2)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceL2)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceL2)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceL2)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMax)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMean)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMean)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMean)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceProd)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceProd)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceProd)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceProd)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceSum)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, int32_t, ReduceSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceLogSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceLogSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceLogSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceSumSquare)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceSumSquare)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceSumSquare)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, float, ReduceLogSumExp)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, double, ReduceLogSumExp)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, MLFloat16, ReduceLogSumExp)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, float, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, double, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, MLFloat16, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, int8_t, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, int16_t, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, int32_t, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, int64_t, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, uint8_t, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, uint16_t, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, uint32_t, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, uint64_t, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, 8, bool, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint8_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint16_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint32_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint64_t, Cast)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, bool, Cast)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 2, 10, float, Pad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 2, 10, double, Pad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 2, 10, MLFloat16, Pad)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 5, Reshape)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 4, Reshape_1)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, Shape)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, Tile)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, Tile)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, Tile)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, Transpose)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, float, InstanceNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, double, InstanceNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 6, MLFloat16, InstanceNormalization)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, RNN)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, RNN)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, RNN)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, GRU)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, GRU)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, GRU)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, float, LSTM)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, double, LSTM)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, MLFloat16, LSTM)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 9, int32_t, Slice)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 9, int64_t, Slice)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 9, float, Slice)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, Compress)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, Flatten)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, float, Upsample)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, double, Upsample)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, MLFloat16, Upsample)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, int32_t, Upsample)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 9, uint8_t, Upsample)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 2, 10, Split)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, ConstantOfShape)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int8_t, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int16_t, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint8_t, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint16_t, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint32_t, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint64_t, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Shrink)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, float, Less)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, double, Less)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 7, 8, MLFloat16, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint32_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint64_t, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, double, Less)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Less)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, EyeLike)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, Scatter)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, MLFloat16, Where)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, Where)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, Where)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, Where)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint8_t, Where)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, bool, NonZero)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, uint8_t, NonZero)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int32_t, NonZero)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, int64_t, NonZero)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, float, NonZero)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 9, TopK)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 8, 8, Scan)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 9, 10, Scan)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, Loop)>,

      // opset 10
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, double, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, Dropout)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, float, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, double, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, NonMaxSuppression)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, float, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, double, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, MLFloat16, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, int32_t, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, uint8_t, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, ReverseSequence)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, float, RoiAlign)>,
      // // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, double, RoiAlign)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, int32_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, int64_t, Slice)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, float, Slice)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, float, ThresholdedRelu)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, double, ThresholdedRelu)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, MLFloat16, ThresholdedRelu)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 10, 10, TopK)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 1, 10, If)>,

      // opset 11
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ArgMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ArgMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ArgMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ArgMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ArgMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ArgMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Compress)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Concat)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Flatten)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Gather)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, GatherElements)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Gemm)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Gemm)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Gemm)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, If)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Loop)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, NonMaxSuppression)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Range)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceL1)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceL1)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceL1)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceL1)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceL2)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceL2)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceL2)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceL2)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceLogSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceLogSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceLogSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceLogSumExp)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceLogSumExp)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceLogSumExp)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceMax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceMean)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceMean)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceMean)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceMean)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceMin)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceProd)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceProd)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceProd)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceProd)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, ReduceSum)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ReduceSumSquare)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ReduceSumSquare)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ReduceSumSquare)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Scan)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, ScatterElements)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, Slice)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int64_t, Slice)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Slice)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Softmax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Softmax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Softmax)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Split)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Squeeze)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, TopK)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, Unsqueeze)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Conv)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Conv)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Conv)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, ConvTranspose)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, ConvTranspose)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, ConvTranspose)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, AveragePool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, MaxPool)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, uint8_t, Resize)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Clip)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Pad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Pad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Pad)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, bool, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int32_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, int64_t, Equal)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, float, Round)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, double, Round)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, MLFloat16, Round)>,
      // BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kHipExecutionProvider, kOnnxDomain, 11, CumSum)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

#ifndef DISABLE_CONTRIB_OPS
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::hip::RegisterHipContribKernels(kernel_registry));
#endif

#ifdef ENABLE_TRAINING
  ORT_RETURN_IF_ERROR(::onnxruntime::hip::RegisterHipTrainingKernels(kernel_registry));
#endif

  return Status::OK();
}

KernelRegistryAndStatus GetHipKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterHipKernels(*ret.kernel_registry);
  return ret;
}

} // namespace hip

std::shared_ptr<KernelRegistry> HIPExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus k = onnxruntime::hip::GetHipKernelRegistry();
  // throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(k.st);
  return k.kernel_registry;
}

HIPExecutionProvider::HIPExecutionProvider(const HIPExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kHipExecutionProvider}, device_id_(info.device_id) {

  HIP_CALL_THROW(hipSetDevice(device_id_));

  // must wait GPU idle, otherwise hipGetDeviceProperties might fail
  HIP_CALL_THROW(hipDeviceSynchronize());
  HIP_CALL_THROW(hipGetDeviceProperties(&prop_, device_id_));

  DeviceAllocatorRegistrationInfo default_memory_info(
      {OrtMemTypeDefault, [](OrtDevice::DeviceId device_id) { return onnxruntime::make_unique<HIPAllocator>(device_id, CUDA); }, std::numeric_limits<size_t>::max()});
  allocator_ = CreateAllocator(default_memory_info, device_id_);
  InsertAllocator(allocator_);


  DeviceAllocatorRegistrationInfo pinned_memory_info(
      {OrtMemTypeCPUOutput, [](OrtDevice::DeviceId device_id) { return onnxruntime::make_unique<HIPPinnedAllocator>(device_id, CUDA_PINNED); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(pinned_memory_info, CPU_ALLOCATOR_DEVICE_ID));

  // TODO: this is actually used for the hip kernels which explicitly ask for inputs from CPU.
  // This will be refactored/removed when allocator and execution provider are decoupled.
  DeviceAllocatorRegistrationInfo cpu_memory_info({OrtMemTypeCPUInput,
                                                   [](int device_id) { return onnxruntime::make_unique<CPUAllocator>(
                                                                           onnxruntime::make_unique<OrtMemoryInfo>(
                                                                               "CUDA_CPU",
                                                                               OrtAllocatorType::OrtDeviceAllocator,
                                                                               OrtDevice(),
                                                                               device_id,
                                                                               OrtMemTypeCPUInput)); },
                                                   std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(cpu_memory_info, CPU_ALLOCATOR_DEVICE_ID));
}

HIPExecutionProvider::~HIPExecutionProvider() {
  auto cpu_alloc = GetAllocator(CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPU);
  std::lock_guard<OrtMutex> lock(deferred_release_cpu_ptr_mutex_);
  auto it = deferred_release_cpu_ptr_.begin();
  while (it != deferred_release_cpu_ptr_.end()) {
    auto& e = it->first;
    auto& v = it->second;
    if (v.recorded)
      HIP_CALL_THROW(hipEventSynchronize(e));
    for (auto p : v.cpu_ptrs) {
      cpu_alloc->Free(p);
    }
    HIP_CALL_THROW(hipEventDestroy(e));
    it = deferred_release_cpu_ptr_.erase(it);
  }
}

AllocatorPtr HIPExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeDefault) {
    return allocator_;
  } else {
    return IExecutionProvider::GetAllocator(id, mem_type);
  }
}

std::unique_ptr<onnxruntime::IDataTransfer> HIPExecutionProvider::GetDataTransfer() const {
  return onnxruntime::make_unique<onnxruntime::GPUDataTransfer>();
}


std::vector<std::unique_ptr<ComputeCapability>>
HIPExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {

  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::unordered_set<const NodeArg*> defs_outside_hip;

  for (auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph_viewer.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    const KernelCreateInfo* hip_kernel_def = nullptr;
    if (!node.GetExecutionProviderType().empty()) {
      defs_outside_hip.insert(node.OutputDefs().cbegin(), node.OutputDefs().cend());
      continue;
    }
    hip_kernel_def = GetKernelRegistry()->TryFindKernel(node, Type());
    if (hip_kernel_def == nullptr) {
      // node is not in hip exeuction provider if no kernel def found,
      // or if other execution provider already assigned to it
      defs_outside_hip.insert(node.OutputDefs().cbegin(), node.OutputDefs().cend());
      continue;
    }

    bool not_supported = false;
    bool force_outside = false;
    bool force_inside = false;  // for some compute heavy ops, we'll force it to run inside HIP

    if (!force_inside && (not_supported || force_outside)) {
      defs_outside_hip.insert(node.OutputDefs().cbegin(), node.OutputDefs().cend());
      if (not_supported) {
        LOGS_DEFAULT(WARNING) << "HIP kernel not supported. Fallback to CPU execution provider for Op type: " << node.OpType() << " node name: " << node.Name();
      } else if (force_outside) {
        LOGS_DEFAULT(INFO) << "Force fallback to CPU execution provider for Op type: " << node.OpType() << " node name: " << node.Name();
      }
    } else {
      // for nodes placed on HIP, check if its output is on CPU
      ORT_THROW_IF_ERROR(node.ForEachWithIndex(
          node.OutputDefs(),
          [&](const NodeArg& def, size_t out_index) {
            if (hip_kernel_def->kernel_def->OutputMemoryType(out_index) != OrtMemTypeDefault)
              defs_outside_hip.insert(&def);
            return Status::OK();
          }));
      std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }
  return result;
}

Status HIPExecutionProvider::Sync() const {
  HIP_RETURN_IF_ERROR(hipDeviceSynchronize());
  return Status::OK();
}

Status HIPExecutionProvider::OnRunStart() {
  // always set HIP device when session::Run() in case it runs in a worker thread
  HIP_RETURN_IF_ERROR(hipSetDevice(GetDeviceId()));
  auto cpu_alloc = GetAllocator(0, OrtMemTypeCPU);
  // check if hipEvents has passed for deferred release
  // note that we need to take a mutex in case of multi-threaded Run()
  std::lock_guard<OrtMutex> lock(deferred_release_cpu_ptr_mutex_);
  auto it = deferred_release_cpu_ptr_.begin();
  while (it != deferred_release_cpu_ptr_.end()) {
    auto& e = it->first;
    auto& v = it->second;
    // note that hipEventQuery returns hipSucess before first hipEventRecord
    if (v.recorded && hipSuccess == hipEventQuery(e)) {
      for (auto p : v.cpu_ptrs) {
        cpu_alloc->Free(p);
      }
      hipEvent_t expired_event = it->first;
      it = deferred_release_cpu_ptr_.erase(it);
      HIP_RETURN_IF_ERROR(hipEventDestroy(expired_event));
    } else {
      ++it;
    }
  }

  auto& current_deferred_release_event = GetPerThreadContext().GetCurrentDeferredReleaseEvent();
  HIP_RETURN_IF_ERROR(hipEventCreate(&current_deferred_release_event));
  deferred_release_cpu_ptr_.emplace(current_deferred_release_event, DeferredReleaseCPUPtrs());

  return Status::OK();
}

Status HIPExecutionProvider::OnRunEnd() {
  // record deferred release event on default stream, and release per_thread_context
  auto current_deferred_release_event = GetPerThreadContext().GetCurrentDeferredReleaseEvent();
  HIP_RETURN_IF_ERROR(hipEventRecord(current_deferred_release_event, nullptr));
  ReleasePerThreadStuffs();
  std::lock_guard<OrtMutex> lock(deferred_release_cpu_ptr_mutex_);
  deferred_release_cpu_ptr_[current_deferred_release_event].recorded = true;
 
  return Status::OK();
}

Status HIPExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& /*fused_nodes*/,
                                     std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return Status(ONNXRUNTIME, NOT_IMPLEMENTED);
}

void HIPExecutionProvider::AddDeferredReleaseCPUPtr(void* p) {
  // when not running in InferenceSession (e.g. Test)
  // it's OK to not remember the deferred release ptr
  // as the actual memory will be cleaned in arena allocator dtor
  auto current_deferred_release_event = GetPerThreadContext().GetCurrentDeferredReleaseEvent();
  if (current_deferred_release_event) {
    std::lock_guard<OrtMutex> lock(deferred_release_cpu_ptr_mutex_);
    auto iter = deferred_release_cpu_ptr_.find(current_deferred_release_event);
    ORT_ENFORCE(iter != deferred_release_cpu_ptr_.end());
    iter->second.cpu_ptrs.push_back(p);
  }
}

HIPExecutionProvider::PerThreadContext& HIPExecutionProvider::GetPerThreadContext() const {
  if (per_thread_context_map_ == nullptr) {
    per_thread_context_map_ = onnxruntime::make_unique<PerThreadContextMap>();
  }

  auto* p = per_thread_context_map_.get();
  if (p->count(this) == 0) {
    std::lock_guard<OrtMutex> lock(context_pool_mutex_);
    std::shared_ptr<PerThreadContext> ptc;
    if (retired_context_pool_.empty()) {
      ptc = std::make_shared<PerThreadContext>(device_id_);
    } else {
      ptc = retired_context_pool_.back();
      retired_context_pool_.pop_back();
    }
    p->insert(std::make_pair(this, ptc));
  }
  return *(p->at(this));
}

void HIPExecutionProvider::ReleasePerThreadStuffs() const {
  ORT_ENFORCE(per_thread_context_map_ != nullptr);
  auto iter_ctx = per_thread_context_map_->find(this);
  ORT_ENFORCE(iter_ctx != per_thread_context_map_->end());

  std::lock_guard<OrtMutex> lock(context_pool_mutex_);
  retired_context_pool_.push_back(iter_ctx->second);
  per_thread_context_map_->erase(iter_ctx);
  // Release TLS if empty to avoid memory leak report
  if (per_thread_context_map_->empty()) {
    per_thread_context_map_.reset(nullptr);
  }
}

thread_local std::unique_ptr<HIPExecutionProvider::PerThreadContextMap> HIPExecutionProvider::per_thread_context_map_;

HIPExecutionProvider::PerThreadContext::PerThreadContext(OrtDevice::DeviceId device_id) {
  HIP_CALL_THROW(hipSetDevice(device_id));
  HIPBLAS_CALL_THROW(hipblasCreate(&hipblas_handle_));
  MIOPEN_CALL_THROW(miopenCreate(&miopen_handle_));
  // CURAND_CALL_THROW(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT));

  DeviceAllocatorRegistrationInfo default_memory_info(
      {OrtMemTypeDefault,
       [](OrtDevice::DeviceId id) { return onnxruntime::make_unique<HIPAllocator>(id, CUDA); }, std::numeric_limits<size_t>::max()});

  allocator_ = CreateAllocator(default_memory_info, device_id);
}

HIPExecutionProvider::PerThreadContext::~PerThreadContext() {
  // dtor shouldn't throw. if something went wrong earlier (e.g. out of HIP memory) the handles
  // here may be bad, and the destroy calls can throw.
  // https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-dtor-noexcept
  try {
    HIPBLAS_CALL(hipblasDestroy(hipblas_handle_));
  } catch (const std::exception& ex) {
    LOGS_DEFAULT(ERROR) << "hipblasDestroy threw:" << ex.what();
  }

  try {
    MIOPEN_CALL(miopenDestroy(miopen_handle_));
  } catch (const std::exception& ex) {
    LOGS_DEFAULT(ERROR) << "miopenDestroy threw:" << ex.what();
  }
  // CURAND_CALL_THROW(curandDestroyGenerator(curand_generator_));
}

}  // namespace onnxruntime

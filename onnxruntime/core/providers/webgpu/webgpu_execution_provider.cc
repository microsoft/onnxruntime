// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_execution_provider.h"

#include <mutex>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#endif

#include "allocator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/fallback_cpu_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/run_options.h"
#include "core/graph/function_utils.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/common/parse_string.h"

#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/external_data_loader.h"
#include "core/providers/webgpu/webgpu_profiler.h"
#include "core/providers/webgpu/tensor/cast.h"
#include "core/providers/webgpu/tensor/expand.h"
#include "core/providers/webgpu/tensor/grid_sample.h"
#include "core/providers/webgpu/generator/range.h"
#include "core/providers/webgpu/tensor/unsqueeze.h"

namespace onnxruntime {

namespace webgpu {
template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    Tensor* Y = ctx->Output(0, X->Shape());
    return Info().GetDataTransferManager().CopyTensor(*X, *Y);
  }
};

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPU, 0)
        .ExecQueueId(0)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPU, 0)
        .ExecQueueId(1)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

#define KERNEL_CREATE_INFO_VERSIONED(Start, End, Op) \
  BuildKernelCreateInfo<                             \
      class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, Start, End, Op)>

#define KERNEL_CREATE_INFO(Start, Op) \
  BuildKernelCreateInfo<              \
      class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, Start, Op)>

#define KERNEL_CREATE_INFO_TYPED(Start, type, Op) \
  BuildKernelCreateInfo<                          \
      class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, Start, type, Op)>

static const BuildKernelCreateInfoFn build_kernel_create_info_function_table[] = {
    BuildKernelCreateInfo<void>,  // default entry to avoid the list becoming empty after ops-reducing
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,

    // element-wise operators
    // unary - math
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Abs),
    KERNEL_CREATE_INFO(13, Abs),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Neg),
    KERNEL_CREATE_INFO(13, Neg),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Floor),
    KERNEL_CREATE_INFO(13, Floor),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Ceil),
    KERNEL_CREATE_INFO(13, Ceil),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Reciprocal),
    KERNEL_CREATE_INFO(13, Reciprocal),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Sqrt),
    KERNEL_CREATE_INFO(13, Sqrt),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Exp),
    KERNEL_CREATE_INFO(13, Exp),
    KERNEL_CREATE_INFO_VERSIONED(9, 12, Erf),
    KERNEL_CREATE_INFO(13, Erf),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Sigmoid),
    KERNEL_CREATE_INFO(13, Sigmoid),
    KERNEL_CREATE_INFO(6, HardSigmoid),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Log),
    KERNEL_CREATE_INFO(13, Log),

    KERNEL_CREATE_INFO(7, Sin),
    KERNEL_CREATE_INFO(7, Cos),
    KERNEL_CREATE_INFO(7, Tan),
    KERNEL_CREATE_INFO(7, Asin),
    KERNEL_CREATE_INFO(7, Acos),
    KERNEL_CREATE_INFO(7, Atan),
    KERNEL_CREATE_INFO(9, Sinh),
    KERNEL_CREATE_INFO(9, Cosh),
    KERNEL_CREATE_INFO(9, Asinh),
    KERNEL_CREATE_INFO(9, Acosh),
    KERNEL_CREATE_INFO(9, Atanh),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Tanh),
    KERNEL_CREATE_INFO(13, Tanh),
    KERNEL_CREATE_INFO(1, Not),

    // // activations
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, float, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, float, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, float, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, MLFloat16, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, Clip)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, MLFloat16, Clip)>,
    KERNEL_CREATE_INFO(6, Elu),
    KERNEL_CREATE_INFO_VERSIONED(6, 12, Relu),
    KERNEL_CREATE_INFO_VERSIONED(13, 13, Relu),
    KERNEL_CREATE_INFO(14, Relu),
    KERNEL_CREATE_INFO_VERSIONED(6, 15, LeakyRelu),
    KERNEL_CREATE_INFO(16, LeakyRelu),
    KERNEL_CREATE_INFO(10, ThresholdedRelu),
    KERNEL_CREATE_INFO(20, Gelu),
    KERNEL_CREATE_INFO_VERSIONED(1, 21, Softplus),
    KERNEL_CREATE_INFO(22, Softplus),

    // // binary - math
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
    KERNEL_CREATE_INFO_VERSIONED(7, 11, Pow),
    KERNEL_CREATE_INFO_VERSIONED(12, 12, Pow),
    KERNEL_CREATE_INFO_VERSIONED(13, 14, Pow),
    KERNEL_CREATE_INFO(15, Pow),
    KERNEL_CREATE_INFO_VERSIONED(7, 10, Equal),
    KERNEL_CREATE_INFO_VERSIONED(11, 12, Equal),
    KERNEL_CREATE_INFO_VERSIONED(13, 18, Equal),
    KERNEL_CREATE_INFO(19, Equal),
    KERNEL_CREATE_INFO_VERSIONED(7, 8, Greater),
    KERNEL_CREATE_INFO_VERSIONED(9, 12, Greater),
    KERNEL_CREATE_INFO(13, Greater),
    KERNEL_CREATE_INFO_VERSIONED(12, 15, GreaterOrEqual),
    KERNEL_CREATE_INFO(16, GreaterOrEqual),
    KERNEL_CREATE_INFO_VERSIONED(7, 8, Less),
    KERNEL_CREATE_INFO_VERSIONED(9, 12, Less),
    KERNEL_CREATE_INFO(13, Less),
    KERNEL_CREATE_INFO_VERSIONED(12, 15, LessOrEqual),
    KERNEL_CREATE_INFO(16, LessOrEqual),
    KERNEL_CREATE_INFO(7, And),

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 14, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 15, 18, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Shape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, Shape)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 5, 12, Reshape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 13, Reshape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, 18, Reshape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Reshape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Reshape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, 24, Reshape)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 25, Reshape)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 13, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, 15, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, 18, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, 23, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 24, 24, Identity)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 25, Identity)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 20, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, 23, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 24, Squeeze)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 19, ReduceMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 20, ReduceMax)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMean)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceMean)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMean)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceMean)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, ReduceMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, ReduceMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 19, ReduceMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 20, ReduceMin)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceProd)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceProd)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceProd)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceProd)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, ReduceSum)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceL1)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceL1)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceL1)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceL1)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceL2)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceL2)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceL2)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceL2)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceLogSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceLogSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceLogSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceLogSum)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceSumSquare)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceSumSquare)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceSumSquare)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceSumSquare)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ReduceLogSumExp)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ReduceLogSumExp)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, ReduceLogSumExp)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ReduceLogSumExp)>,

    KERNEL_CREATE_INFO_VERSIONED(9, 15, Where),
    KERNEL_CREATE_INFO(16, Where),

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, Transpose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 20, Transpose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Transpose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, 23, Transpose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 24, Transpose)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, DepthToSpace)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, DepthToSpace)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 12, DepthToSpace)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 13, DepthToSpace)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 21, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 22, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 10, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 21, Conv)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 22, Conv)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ConvTranspose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, ConvTranspose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 10, ConvTranspose)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, ConvTranspose)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 7, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 8, 9, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 7, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 8, 9, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 11, MaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 12, MaxPool)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 9, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 7, 9, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, AveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, AveragePool)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, GlobalAveragePool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, GlobalAveragePool)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, GlobalMaxPool)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, GlobalMaxPool)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 8, Gemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 10, Gemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Gemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Gemm)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 12, MatMul)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, MatMul)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ArgMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ArgMax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, ArgMax)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, ArgMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ArgMin)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, ArgMin)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Softmax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Softmax)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Softmax)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 3, Concat)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Concat)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Concat)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 1, Split)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 2, 10, Split)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Split)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Split)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, Split)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, Gather)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Gather)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Gather)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, GatherElements)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, GatherElements)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 11, GatherND)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, 12, GatherND)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, GatherND)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, Resize)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Resize)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Resize)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 18, Resize)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, Resize)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 10, 10, Resize)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 11, 12, Resize)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 13, 17, Resize)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 18, 18, Resize)>,
    // BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 19, Resize)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 9, Slice)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 10, 10, Slice)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Slice)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Slice)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 8, Flatten)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 10, Flatten)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Flatten)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 20, Flatten)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, Flatten)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 12, Tile)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, Tile)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 17, LayerNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, RMSNormalization)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, RotaryEmbedding)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 21, LpNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 22, LpNormalization)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 5, InstanceNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 1, 5, InstanceNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 6, 21, InstanceNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 6, 21, InstanceNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 22, InstanceNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 22, InstanceNormalization)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 12, Einsum)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 2, 10, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 17, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, 18, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, 22, Pad)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 23, Pad)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 1, 10, If)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, If)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 18, If)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 19, 20, If)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 21, If)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 7, 8, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 9, 13, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, 14, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 15, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 7, 8, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 9, 13, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 14, 14, BatchNormalization)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kMSInternalNHWCDomain, 15, BatchNormalization)>,

    KERNEL_CREATE_INFO_VERSIONED(10, 12, DequantizeLinear),
    KERNEL_CREATE_INFO_VERSIONED(13, 18, DequantizeLinear),
    KERNEL_CREATE_INFO_VERSIONED(19, 20, DequantizeLinear),
    KERNEL_CREATE_INFO_VERSIONED(21, 22, DequantizeLinear),
    KERNEL_CREATE_INFO(23, DequantizeLinear),

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 13, CumSum)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 14, CumSum)>,

    KERNEL_CREATE_INFO_VERSIONED(1, 9, TopK),
    KERNEL_CREATE_INFO_VERSIONED(10, 10, TopK),
    KERNEL_CREATE_INFO_VERSIONED(11, 23, TopK),
    KERNEL_CREATE_INFO(24, TopK),

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ScatterND)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 15, ScatterND)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, 17, ScatterND)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ScatterND)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 11, 12, ScatterElements)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 13, 15, ScatterElements)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, 17, ScatterElements)>,
    BuildKernelCreateInfo<class ONNX_OPERATOR_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 18, ScatterElements)>,

    BuildKernelCreateInfo<class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kWebGpuExecutionProvider, kOnnxDomain, 16, 19, GridSample)>,
};

std::unique_ptr<KernelRegistry> RegisterKernels(bool enable_graph_capture, bool enable_int64) {
  auto kernel_registry = std::make_unique<KernelRegistry>();

  for (auto& function_table_entry : build_kernel_create_info_function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_THROW_IF_ERROR(kernel_registry->Register(std::move(info)));
    }
  }

  // Register Cast kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastKernelInfo<6, 8>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastKernelInfo<9, 12>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastKernelInfo<13, 18>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastKernelInfo<19, 20>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastKernelInfo<21, 22>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateCastKernelInfo<23>(enable_int64)));

  // Register Range kernels with conditional int64 support
  RegisterRangeKernels(*kernel_registry, enable_int64);

  // Register Unsqueeze kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo<1, 10>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo<11, 12>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo<13, 20>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo<21, 22>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo<23, 23>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeVersionedKernelInfo<24, 24>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateUnsqueezeKernelInfo<25>(enable_int64)));

  // Register Expand kernels with conditional int64 support
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateExpandVersionedKernelInfo<8, 12>(enable_int64)));
  ORT_THROW_IF_ERROR(kernel_registry->Register(CreateExpandKernelInfo<13>(enable_int64)));

#ifndef DISABLE_CONTRIB_OPS
  Status status = ::onnxruntime::contrib::webgpu::RegisterWebGpuContribKernels(*kernel_registry, enable_graph_capture);
  ORT_ENFORCE(status.IsOK(), "Failed to register WebGPU contrib kernels: " + status.ErrorMessage());
#endif

  return kernel_registry;
}

#if defined(ORT_USE_EP_API_ADAPTERS)

namespace {
std::mutex g_kernel_registry_mutex;
std::shared_ptr<KernelRegistry> g_kernel_registry;
std::shared_ptr<KernelRegistry> g_graph_capture_kernel_registry;
std::shared_ptr<KernelRegistry> g_int64_kernel_registry;
}  // namespace

void CleanupKernelRegistries() {
  std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
  g_kernel_registry.reset();
  g_graph_capture_kernel_registry.reset();
  g_int64_kernel_registry.reset();
}
#endif

std::shared_ptr<KernelRegistry> GetKernelRegistry(bool enable_graph_capture, bool enable_int64) {
  // kernel registry variables are defined differently based on build configuration
  //
  // - When building as a static library, use static local variable. This is because
  //   we don't have a reliable way to explicitly destroy the kernel registry after
  //   use.
  //
  // - When building as a shared library, use global variables. The cleanup will be performed
  //   when `ReleaseEpFactory` is called.
  //
  // Graph capture mode needs a separate kernel registry because contrib kernel registration
  // differs based on enable_graph_capture, and enable_int64 is always true when
  // enable_graph_capture is true.
  if (enable_graph_capture) {
#if !defined(ORT_USE_EP_API_ADAPTERS)
    static std::shared_ptr<KernelRegistry> registry = RegisterKernels(true, true);
    return registry;
#else
    std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
    if (g_graph_capture_kernel_registry == nullptr) {
      g_graph_capture_kernel_registry = RegisterKernels(true, true);
    }
    return g_graph_capture_kernel_registry;
#endif
  } else if (enable_int64) {
#if defined(ORT_USE_EP_API_ADAPTERS)
    std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
    if (g_int64_kernel_registry == nullptr) {
      g_int64_kernel_registry = RegisterKernels(false, true);
    }
    return g_int64_kernel_registry;
#else
    static std::shared_ptr<KernelRegistry> registry = RegisterKernels(false, true);
    return registry;
#endif
  } else {
#if defined(ORT_USE_EP_API_ADAPTERS)
    std::lock_guard<std::mutex> lock(g_kernel_registry_mutex);
    if (g_kernel_registry == nullptr) {
      g_kernel_registry = RegisterKernels(false, false);
    }
    return g_kernel_registry;
#else
    static std::shared_ptr<KernelRegistry> registry = RegisterKernels(false, false);
    return registry;
#endif
  }
}

}  // namespace webgpu

using namespace webgpu;

WebGpuExecutionProvider::WebGpuExecutionProvider(int context_id,
                                                 WebGpuContext& context,
                                                 WebGpuExecutionProviderConfig&& config)
    : IExecutionProvider{kWebGpuExecutionProvider, WebGpuDevice},
      context_id_{context_id},
      context_{context},
      preferred_data_layout_{config.data_layout},
      force_cpu_node_names_{std::move(config.force_cpu_node_names)},
      enable_graph_capture_{config.enable_graph_capture},
      // enable_int64_ is always true when enable_graph_capture_ is true
      enable_int64_{config.enable_graph_capture || config.enable_int64},
      multi_rotary_cache_concat_offset_{config.multi_rotary_cache_concat_offset},
      prepack_allocator_{std::make_shared<webgpu::GpuBufferAllocator>(context_.InitializerBufferManager(), false)} {
  // If graph capture is enabled, create a dedicated buffer manager for graph mode
  if (enable_graph_capture_) {
    // Create buffer manager for graph capture mode with appropriate cache modes
    graph_buffer_mgr_ = webgpu::BufferManagerFactory::Create(
        context_,
        webgpu::BufferCacheMode::Graph,
        webgpu::BufferCacheMode::GraphSimple,
        webgpu::BufferCacheMode::Disabled);
  }

  if (config.enable_pix_capture) {
#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
    // set pix frame generator
    pix_frame_generator_ = context_.CreatePIXFrameGenerator();
#else
    ORT_THROW("Support PIX capture requires extra build flags (--enable_pix_capture)");
#endif  // ENABLE_PIX_FOR_WEBGPU_EP
  }
}

std::vector<AllocatorPtr> WebGpuExecutionProvider::CreatePreferredAllocators() {
  return {
      // allocator for initializers
      std::make_unique<webgpu::GpuBufferAllocator>(context_.InitializerBufferManager(), true),
      // default allocator
      std::make_unique<webgpu::GpuBufferAllocator>(BufferManager(), false),
  };
}

#if !defined(ORT_USE_EP_API_ADAPTERS)
std::vector<std::unique_ptr<ComputeCapability>> WebGpuExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph,
    const IKernelLookup& kernel_lookup,
    const GraphOptimizerRegistry& /* graph_optimizer_registry */,
    IResourceAccountant* /* resource_accountant */) const {
  InlinedVector<NodeIndex> candidates;
  // `tenative_candidates` is a subset of `candidates`.
  InlinedVector<NodeIndex> tenative_candidates;
  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      // If the node was added by layout transformer, do not move it to CPU
      if (node.GetExecutionProviderType() == kWebGpuExecutionProvider) {
        candidates.push_back(node.Index());
      }
      continue;
    }

    const KernelCreateInfo* webgpu_kernel_def = kernel_lookup.LookUpKernel(node);
    // none of the provided registries has a webgpu kernel for this node
    if (webgpu_kernel_def == nullptr) {
      LOGS(*GetLogger(), INFO) << "webgpu kernel not found in registries for Op type: "
                               << node.OpType() << " node name: " << node.Name();
      continue;
    }

    // TODO: currently this lookup is O(N). If the list becomes large we should optimize this.
    if (std::find(force_cpu_node_names_.cbegin(),
                  force_cpu_node_names_.cend(),
                  node.Name()) != force_cpu_node_names_.cend()) {
      LOGS(*GetLogger(), INFO) << "Force CPU execution for node: " << node.Name();
      continue;
    }

    //
    // The following code checks if the node is really supported by webgpu EP
    //

#define FALLBACK_TO_CPU_IF_EXIST_INPUT(idx)           \
  if (inputs.size() > idx && inputs[idx]->Exists()) { \
    continue;                                         \
  }

#define FALLBACK_TO_CPU_IF_EXIST_OUTPUT(idx)            \
  if (outputs.size() > idx && outputs[idx]->Exists()) { \
    continue;                                           \
  }

    // Check for Attention
    if (node.OpType() == "Attention" && node.Domain() == kMSDomain) {
      const auto& inputs = node.InputDefs();
      const auto& outputs = node.OutputDefs();

      // Current implementation does not support mask_index(input[3]), past(input[4]) and past_seq_len(input[6])
      FALLBACK_TO_CPU_IF_EXIST_INPUT(3);
      FALLBACK_TO_CPU_IF_EXIST_INPUT(4);
      FALLBACK_TO_CPU_IF_EXIST_INPUT(6);

      // Current implementation does not support present(output[1])
      FALLBACK_TO_CPU_IF_EXIST_OUTPUT(1);

      // If attribute past_present_share_buffer is set, fallback to CPU
      const auto& past_present_share_buffer = node.GetAttributes().find("past_present_share_buffer");
      if (past_present_share_buffer != node.GetAttributes().end() &&
          past_present_share_buffer->second.i() != 0) {
        continue;
      }
    }

    candidates.push_back(node.Index());
    tenative_candidates.push_back(node.Index());
  }

  auto cpu_nodes = GetCpuPreferredNodes(graph, kernel_lookup, tenative_candidates, *GetLogger());
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    if (cpu_nodes.contains(node_index)) {
      continue;
    }

    auto sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node_index);
    result.emplace_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }
  return result;
}

#endif  // !defined(ORT_USE_EP_API_ADAPTERS)

std::unique_ptr<onnxruntime::IDataTransfer> WebGpuExecutionProvider::GetDataTransfer() const {
  return std::make_unique<webgpu::DataTransfer>(BufferManager());
}

#if defined(__wasm__)
std::unique_ptr<onnxruntime::IExternalDataLoader> WebGpuExecutionProvider::GetExternalDataLoader() const {
  return std::make_unique<webgpu::ExternalDataLoader>();
}
#endif

std::optional<bool> WebGpuExecutionProvider::ShouldConvertDataLayoutForOp(std::string_view node_domain,
                                                                          std::string_view node_op_type,
                                                                          DataLayout target_data_layout) const {
  // NHWC for Resize operator is not implemented on kWebGpuExecutionProvider
  if (node_domain == kOnnxDomain && node_op_type == "Resize") {
    return target_data_layout != DataLayout::NHWC;
  }

  // GridSample is NCHW-only (opset 16 spec requires NCHW input)
  if (node_domain == kOnnxDomain && node_op_type == "GridSample") {
    return target_data_layout != DataLayout::NHWC;
  }

  // WebGPU perfer NCHW for InstanceNormalization due to a better performance
  if (node_domain == kOnnxDomain && node_op_type == "InstanceNormalization") {
    return target_data_layout != DataLayout::NHWC;
  }

  return std::nullopt;
}

WebGpuExecutionProvider::~WebGpuExecutionProvider() {
  // Release all resources associated with the captured graph
  if (!captured_commands_.empty()) {
    context_.ReleaseGraphResources(captured_commands_);
  }
  // The graph_buffer_mgr_ will be automatically cleaned up by unique_ptr

  WebGpuContextFactory::ReleaseContext(context_id_);
}

std::unique_ptr<profiling::EpProfiler> WebGpuExecutionProvider::GetProfiler() {
  auto profiler = std::make_unique<WebGpuProfiler>(context_);
  // Only set session_profiler_ on the first call (session-level profiler).
  // Subsequent calls from run-level profiling create temporary profilers that
  // should not overwrite it, as those profilers have shorter lifetimes.
  if (session_profiler_ == nullptr) {
    session_profiler_ = profiler.get();
  }
  return profiler;
}

Status WebGpuExecutionProvider::OnRunStart(const onnxruntime::RunOptions& run_options) {
  if (context_.ValidationMode() >= ValidationMode::Basic) {
    context_.PushErrorScope();
  }

  // Start profiling if session-level or run-level profiling is enabled
  if (run_options.enable_profiling || (session_profiler_ && session_profiler_->Enabled())) {
    context_.StartProfiling();
  }

  if (IsGraphCaptureEnabled()) {
    auto graph_annotation_str = run_options.config_options.GetConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation);
    int graph_annotation_id = 0;
    if (graph_annotation_str.has_value()) {
      ORT_ENFORCE(onnxruntime::TryParseStringWithClassicLocale<int>(*graph_annotation_str, graph_annotation_id),
                  "Failed to parse the graph annotation id: ",
                  *graph_annotation_str);
    }

    if (graph_annotation_id != -1 && IsGraphCaptureAllowed() && !IsGraphCaptured(graph_annotation_id)) {
      context_.CaptureBegin(&captured_commands_, *graph_buffer_mgr_);
    }
    m_current_graph_annotation_id = graph_annotation_id;
  }

  return Status::OK();
}

Status WebGpuExecutionProvider::OnRunEnd(bool /* sync_stream */, const onnxruntime::RunOptions& run_options) {
  context_.Flush(BufferManager());

  if (IsGraphCaptureEnabled() && !IsGraphCaptured(m_current_graph_annotation_id)) {
    if (m_current_graph_annotation_id != -1 && IsGraphCaptureAllowed()) {
      context_.CaptureEnd();
      is_graph_captured_ = true;
      ORT_RETURN_IF_ERROR(ReplayGraph(m_current_graph_annotation_id));
    } else {
      IncrementRegularRunCountBeforeGraphCapture();
    }
  }

  if (session_profiler_ && session_profiler_->Enabled()) {
    // Session-level profiling: collect into profiler's own events storage.
    context_.CollectProfilingData(session_profiler_->GpuEvents());
  } else if (run_options.enable_profiling) {
    // Run-level profiling: collect into shared events vector.
    context_.CollectProfilingData();
  }

#if defined(ENABLE_PIX_FOR_WEBGPU_EP)
  if (pix_frame_generator_) {
    pix_frame_generator_->GeneratePIXFrame();
  }
#endif  // ENABLE_PIX_FOR_WEBGPU_EP

  if (context_.ValidationMode() >= ValidationMode::Basic) {
    return context_.PopErrorScope();
  } else {
    return Status::OK();
  }
}

bool WebGpuExecutionProvider::IsGraphCaptureEnabled() const {
  return enable_graph_capture_;
}

bool WebGpuExecutionProvider::IsGraphCaptured(int graph_annotation_id) const {
  return is_graph_captured_ && graph_annotation_id != -1;
}

Status WebGpuExecutionProvider::ReplayGraph(int graph_annotation_id) {
  ORT_ENFORCE(IsGraphCaptured(graph_annotation_id));
  // TODO: enable profiling in run level
  if (session_profiler_ && session_profiler_->Enabled()) {
    context_.StartProfiling();
  }
  context_.Replay(captured_commands_, *graph_buffer_mgr_);
  if (session_profiler_ && session_profiler_->Enabled()) {
    // Session-level profiling: collect into profiler's own events storage.
    context_.CollectProfilingData(session_profiler_->GpuEvents());
  }
  return Status::OK();
}

webgpu::BufferManager& WebGpuExecutionProvider::BufferManager() const {
  if (graph_buffer_mgr_) {
    return *graph_buffer_mgr_;
  } else {
    return context_.BufferManager();
  }
}

bool WebGpuExecutionProvider::IsGraphCaptureAllowed() const {
  return regular_run_count_before_graph_capture_ >= min_num_runs_before_cuda_graph_capture_;
}

void WebGpuExecutionProvider::IncrementRegularRunCountBeforeGraphCapture() {
  ++regular_run_count_before_graph_capture_;
}
}  // namespace onnxruntime

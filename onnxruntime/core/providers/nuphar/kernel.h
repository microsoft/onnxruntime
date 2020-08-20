// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"
#include "core/codegen/passes/utils/codegen_context.h"
#include "core/graph/graph.h"
#include "core/providers/nuphar/common/nuphar_subgraph.h"
#include "core/providers/nuphar/compiler/nuphar_compiler.h"
#include "core/providers/nuphar/compiler/initializer_info.h"
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/exec_block.h"

#include <tvm/build_module.h>

#include <map>
#include <unordered_map>

namespace onnxruntime {

class NupharExecutionProvider;

namespace nuphar {

class NupharKernelState;
using NupharFuncStateToComputeCtxMap =
    std::unordered_map<const NupharKernelState*, std::unique_ptr<KernelComputeCtx>>;

class NupharKernelState {
 public:
  explicit NupharKernelState(
      const Node& fused_node,
      const ComputeContext& ctx,
      const NupharExecutionProvider& provider);

  ~NupharKernelState();

  Status Compute(OpKernelContext* op_kernel_context) const;

  void Compile(const NupharSubgraphUnit& subgraph);

  void BuildExecBlocksAndCalls(const std::vector<NupharSubgraphUnit>& subgraphs);

 private:
  const NupharExecutionProvider& provider_;

  // A owner of generated Tensor for weight layout for now
  // TODO: remove it after weight layout refactoring
  std::unordered_map<std::string, std::unique_ptr<Tensor>> generated_initializers_;

  Status codegen_status_;

  // Hold Partition_info for codegen
  std::unique_ptr<OrtSubgraphAllocationInfo> partition_info_;

  // Hold NupharFuncInfo from codegen.
  std::vector<std::unique_ptr<NupharFuncInfo>> func_infos_;

  // ExecBlocks of runtime
  // Ownership of ExecBlock
  std::vector<std::unique_ptr<ExecBlock>> exec_blocks_;

  // Calls
  std::vector<ExecBlock*> exec_block_calls_;

  // Here ComputeContext of Ort is used for allocator
  ComputeContext ctx_;  // the compute context from IExecutionProvider::Compile interface

  static thread_local std::unique_ptr<NupharFuncStateToComputeCtxMap> nuphar_compute_ctx_map_;
};

#define DISABLE_MACRO(X)

#define LIST_NUPHAR_OPS()                                                                   \
  NUPHAR_OP(Abs, 6, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_OP(Add, 7, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(ArgMax, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())               \
  NUPHAR_VERSIONED_OP(ArgMax, 11, 11, DataTypeImpl::AllFixedSizeTensorTypes())              \
  NUPHAR_VERSIONED_OP(ArgMin, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())               \
  NUPHAR_VERSIONED_OP(ArgMin, 11, 11, DataTypeImpl::AllFixedSizeTensorTypes())              \
  NUPHAR_VERSIONED_OP(AveragePool, 7, 9, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes()) \
  NUPHAR_OP(AveragePool, 10, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())             \
  NUPHAR_OP(AveragePool, 11, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())             \
  NUPHAR_OP(Ceil, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_VERSIONED_OP(Clip, 6, 10, DataTypeImpl::AllIEEEFloatTensorTypes())                 \
  NUPHAR_VERSIONED_OP(Clip, 11, 11, DataTypeImpl::AllIEEEFloatTensorTypes())                \
  NUPHAR_VERSIONED_OP(Concat, 4, 10, DataTypeImpl::AllFixedSizeTensorTypes())               \
  NUPHAR_OP(Concat, 11, DataTypeImpl::AllFixedSizeTensorTypes())                            \
  DISABLE_MACRO(NUPHAR_OP(Conv, 1, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes()))      \
  NUPHAR_OP(Crop, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_OP(Div, 7, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_OP(Dropout, 7, DataTypeImpl::AllFixedSizeTensorTypes())                            \
  NUPHAR_OP(Elu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Equal, 7, 10, DataTypeImpl::AllFixedSizeTensorTypes())                \
  NUPHAR_OP(Equal, 11, DataTypeImpl::AllFixedSizeTensorTypes())                             \
  NUPHAR_OP(Erf, 9, DataTypeImpl::GetTensorType<float>())                                   \
  NUPHAR_OP(Exp, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_OP(Expand, 8, DataTypeImpl::AllFixedSizeTensorTypes())                             \
  NUPHAR_VERSIONED_OP(Flatten, 1, 8, DataTypeImpl::AllIEEEFloatTensorTypes())               \
  NUPHAR_VERSIONED_OP(Flatten, 9, 10, DataTypeImpl::AllIEEEFloatTensorTypes())              \
  NUPHAR_OP(Flatten, 11, DataTypeImpl::AllIEEEFloatTensorTypes())                           \
  NUPHAR_OP(Floor, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                              \
  NUPHAR_VERSIONED_OP(Gemm, 7, 8, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())        \
  NUPHAR_VERSIONED_OP(Gemm, 9, 10, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())       \
  NUPHAR_OP(Gemm, 11, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())                    \
  NUPHAR_OP(GlobalAveragePool, 1, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())        \
  NUPHAR_OP(GlobalMaxPool, 1, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())            \
  NUPHAR_OP(Greater, 9, DataTypeImpl::AllFixedSizeTensorTypes())                            \
  NUPHAR_OP(HardSigmoid, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                        \
  NUPHAR_OP(Identity, 1, DataTypeImpl::AllFixedSizeTensorTypes())                           \
  NUPHAR_OP(LeakyRelu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                          \
  NUPHAR_OP(Less, 9, DataTypeImpl::AllFixedSizeTensorTypes())                               \
  NUPHAR_OP(Log, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_VERSIONED_OP(LogSoftmax, 1, 10, DataTypeImpl::AllIEEEFloatTensorTypes())           \
  NUPHAR_OP(LogSoftmax, 11, DataTypeImpl::AllIEEEFloatTensorTypes())                        \
  DISABLE_MACRO(NUPHAR_OP(LSTM, 7, DataTypeImpl::AllIEEEFloatTensorTypes()))                \
  NUPHAR_VERSIONED_OP(MatMul, 1, 8, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())      \
  NUPHAR_OP(MatMul, 9, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())                   \
  NUPHAR_OP(Max, 8, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(MaxPool, 1, 7, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())     \
  NUPHAR_VERSIONED_OP(MaxPool, 8, 9, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())     \
  NUPHAR_OP(MaxPool, 10, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())                 \
  NUPHAR_OP(MaxPool, 11, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())                 \
  NUPHAR_OP(Min, 8, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_OP(Mul, 7, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_OP(Neg, 6, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_OP(Pad, 2, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_OP(ParametricSoftplus, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                 \
  NUPHAR_OP(Pow, 7, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_OP(PRelu, 7, DataTypeImpl::AllIEEEFloatTensorTypes())                              \
  NUPHAR_OP(Relu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_OP(Reciprocal, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                         \
  NUPHAR_VERSIONED_OP(ReduceL1, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())             \
  NUPHAR_OP(ReduceL1, 11, DataTypeImpl::AllFixedSizeTensorTypes())                          \
  NUPHAR_VERSIONED_OP(ReduceL2, 1, 10, DataTypeImpl::AllIEEEFloatTensorTypes())             \
  NUPHAR_OP(ReduceL2, 11, DataTypeImpl::AllIEEEFloatTensorTypes())                          \
  NUPHAR_VERSIONED_OP(ReduceLogSum, 1, 10, DataTypeImpl::AllIEEEFloatTensorTypes())         \
  NUPHAR_OP(ReduceLogSum, 11, DataTypeImpl::AllIEEEFloatTensorTypes())                      \
  NUPHAR_VERSIONED_OP(ReduceLogSumExp, 1, 10, DataTypeImpl::AllIEEEFloatTensorTypes())      \
  NUPHAR_OP(ReduceLogSumExp, 11, DataTypeImpl::AllIEEEFloatTensorTypes())                   \
  NUPHAR_VERSIONED_OP(ReduceMax, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())            \
  NUPHAR_OP(ReduceMax, 11, DataTypeImpl::AllFixedSizeTensorTypes())                         \
  NUPHAR_VERSIONED_OP(ReduceMean, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())           \
  NUPHAR_OP(ReduceMean, 11, DataTypeImpl::AllFixedSizeTensorTypes())                        \
  NUPHAR_VERSIONED_OP(ReduceMin, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())            \
  NUPHAR_OP(ReduceMin, 11, DataTypeImpl::AllFixedSizeTensorTypes())                         \
  NUPHAR_VERSIONED_OP(ReduceProd, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())           \
  NUPHAR_OP(ReduceProd, 11, DataTypeImpl::AllFixedSizeTensorTypes())                        \
  NUPHAR_VERSIONED_OP(ReduceSum, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())            \
  NUPHAR_OP(ReduceSum, 11, DataTypeImpl::AllFixedSizeTensorTypes())                         \
  NUPHAR_VERSIONED_OP(ReduceSumSquare, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())      \
  NUPHAR_OP(ReduceSumSquare, 11, DataTypeImpl::AllFixedSizeTensorTypes())                   \
  NUPHAR_OP(Reshape, 5, DataTypeImpl::AllFixedSizeTensorTypes())                            \
  NUPHAR_OP(ScaledTanh, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                         \
  NUPHAR_OP(Selu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_OP(Shape, 1, DataTypeImpl::AllFixedSizeTensorTypes())                              \
  NUPHAR_OP(Sigmoid, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                            \
  NUPHAR_VERSIONED_OP(Slice, 1, 9, DataTypeImpl::AllFixedSizeTensorTypes())                 \
  NUPHAR_OP(Slice, 10, DataTypeImpl::AllFixedSizeTensorTypes())                             \
  NUPHAR_OP(Slice, 11, DataTypeImpl::AllFixedSizeTensorTypes())                             \
  NUPHAR_VERSIONED_OP(Softmax, 1, 10, DataTypeImpl::AllIEEEFloatTensorTypes())              \
  NUPHAR_OP(Softmax, 11, DataTypeImpl::AllIEEEFloatTensorTypes())                           \
  NUPHAR_OP(Softplus, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                           \
  NUPHAR_OP(Softsign, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                           \
  NUPHAR_VERSIONED_OP(Split, 2, 10, DataTypeImpl::AllIEEEFloatTensorTypes())                \
  NUPHAR_OP(Split, 11, DataTypeImpl::AllIEEEFloatTensorTypes())                             \
  NUPHAR_VERSIONED_OP(Squeeze, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())              \
  NUPHAR_OP(Squeeze, 11, DataTypeImpl::AllFixedSizeTensorTypes())                           \
  NUPHAR_OP(Sqrt, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_OP(Sub, 7, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_OP(Sum, 8, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_OP(Tanh, 6, DataTypeImpl::AllFixedSizeTensorTypes())                               \
  NUPHAR_OP(ThresholdedRelu, 1, DataTypeImpl::AllFixedSizeTensorTypes())                    \
  NUPHAR_OP(Tile, 6, DataTypeImpl::AllFixedSizeTensorTypes())                               \
  NUPHAR_OP(Transpose, 1, DataTypeImpl::AllFixedSizeTensorTypes())                          \
  NUPHAR_VERSIONED_OP(Unsqueeze, 1, 10, DataTypeImpl::AllFixedSizeTensorTypes())            \
  NUPHAR_OP(Unsqueeze, 11, DataTypeImpl::AllFixedSizeTensorTypes())                         \
  NUPHAR_OP(Where, 9, DataTypeImpl::AllFixedSizeTensorTypes())

}  // namespace nuphar
}  // namespace onnxruntime

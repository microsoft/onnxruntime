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

#define LIST_NUPHAR_OPS()                                                                    \
  NUPHAR_VERSIONED_OP(Abs, 6, 12, DataTypeImpl::AllFixedSizeTensorTypes())                   \
  NUPHAR_OP(Abs, 13, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Add, 7, 13, DataTypeImpl::AllFixedSizeTensorTypes())                   \
  NUPHAR_OP(Add, 14, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(ArgMax, 1, 11, DataTypeImpl::AllFixedSizeTensorTypes())                \
  NUPHAR_VERSIONED_OP(ArgMin, 1, 11, DataTypeImpl::AllFixedSizeTensorTypes())                \
  NUPHAR_VERSIONED_OP(AveragePool, 7, 10, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes()) \
  NUPHAR_OP(AveragePool, 11, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())              \
  NUPHAR_VERSIONED_OP(Ceil, 6, 12, DataTypeImpl::AllIEEEFloatTensorTypes())                  \
  NUPHAR_OP(Ceil, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_VERSIONED_OP(Clip, 6, 12, DataTypeImpl::AllIEEEFloatTensorTypes())                  \
  NUPHAR_OP(Clip, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_VERSIONED_OP(Concat, 4, 12, DataTypeImpl::AllFixedSizeTensorTypes())                \
  NUPHAR_OP(Concat, 13, DataTypeImpl::AllFixedSizeTensorTypes())                             \
  DISABLE_MACRO(NUPHAR_OP(Conv, 1, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes()))       \
  NUPHAR_OP(Crop, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Div, 7, 13, DataTypeImpl::AllFixedSizeTensorTypes())                   \
  NUPHAR_OP(Div, 14, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Dropout, 7, 12, DataTypeImpl::AllFixedSizeTensorTypes())               \
  NUPHAR_OP(Dropout, 13, DataTypeImpl::AllFixedSizeTensorTypes())                            \
  NUPHAR_OP(Elu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                 \
  NUPHAR_VERSIONED_OP(Equal, 7, 12, DataTypeImpl::AllFixedSizeTensorTypes())                 \
  NUPHAR_OP(Equal, 13, DataTypeImpl::AllFixedSizeTensorTypes())                              \
  NUPHAR_VERSIONED_OP(Erf, 9, 12, DataTypeImpl::GetTensorType<float>())                      \
  NUPHAR_OP(Erf, 13, DataTypeImpl::GetTensorType<float>())                                   \
  NUPHAR_VERSIONED_OP(Exp, 6, 12, DataTypeImpl::AllIEEEFloatTensorTypes())                   \
  NUPHAR_OP(Exp, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Expand, 8, 12, DataTypeImpl::AllFixedSizeTensorTypes())                \
  NUPHAR_OP(Expand, 13, DataTypeImpl::AllFixedSizeTensorTypes())                             \
  NUPHAR_VERSIONED_OP(Flatten, 1, 12, DataTypeImpl::AllIEEEFloatTensorTypes())               \
  NUPHAR_OP(Flatten, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                            \
  NUPHAR_VERSIONED_OP(Floor, 6, 12, DataTypeImpl::AllIEEEFloatTensorTypes())                 \
  NUPHAR_OP(Floor, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                              \
  NUPHAR_VERSIONED_OP(Gemm, 7, 12, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())        \
  NUPHAR_OP(Gemm, 13, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())                     \
  NUPHAR_OP(GlobalAveragePool, 1, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())         \
  NUPHAR_OP(GlobalMaxPool, 1, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())             \
  NUPHAR_VERSIONED_OP(Greater, 9, 12, DataTypeImpl::AllFixedSizeTensorTypes())               \
  NUPHAR_OP(Greater, 13, DataTypeImpl::AllFixedSizeTensorTypes())                            \
  NUPHAR_OP(HardSigmoid, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                         \
  NUPHAR_VERSIONED_OP(Identity, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())              \
  NUPHAR_OP(Identity, 13, DataTypeImpl::AllFixedSizeTensorTypes())                           \
  NUPHAR_OP(LeakyRelu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                           \
  NUPHAR_VERSIONED_OP(Less, 7, 12, DataTypeImpl::AllFixedSizeTensorTypes())                  \
  NUPHAR_OP(Less, 13, DataTypeImpl::AllFixedSizeTensorTypes())                               \
  NUPHAR_VERSIONED_OP(Log, 6, 12, DataTypeImpl::AllIEEEFloatTensorTypes())                   \
  NUPHAR_OP(Log, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_VERSIONED_OP(LogSoftmax, 1, 12, DataTypeImpl::AllIEEEFloatTensorTypes())            \
  DISABLE_MACRO(NUPHAR_OP(LSTM, 7, DataTypeImpl::AllIEEEFloatTensorTypes()))                 \
  NUPHAR_VERSIONED_OP(MatMul, 1, 12, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())      \
  NUPHAR_OP(MatMul, 13, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())                   \
  NUPHAR_VERSIONED_OP(Max, 8, 12, DataTypeImpl::AllFixedSizeTensorTypes())                   \
  NUPHAR_OP(Max, 13, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(MaxPool, 1, 11, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())     \
  NUPHAR_OP(MaxPool, 12, DataTypeImpl::AllIEEEFloatTensorExceptHalfTypes())                  \
  NUPHAR_VERSIONED_OP(Min, 6, 12, DataTypeImpl::AllFixedSizeTensorTypes())                   \
  NUPHAR_OP(Min, 13, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Mul, 7, 13, DataTypeImpl::AllFixedSizeTensorTypes())                   \
  NUPHAR_OP(Mul, 14, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Neg, 6, 12, DataTypeImpl::AllFixedSizeTensorTypes())                   \
  NUPHAR_OP(Neg, 13, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Pad, 2, 10, DataTypeImpl::AllIEEEFloatTensorTypes())                   \
  NUPHAR_OP(ParametricSoftplus, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                  \
  NUPHAR_VERSIONED_OP(Pow, 7, 12, DataTypeImpl::AllIEEEFloatTensorTypes())                   \
  NUPHAR_OP(Pow, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_VERSIONED_OP(PRelu, 7, 8, DataTypeImpl::AllIEEEFloatTensorTypes())                  \
  NUPHAR_OP(PRelu, 9, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_VERSIONED_OP(Relu, 6, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                  \
  NUPHAR_OP(Relu, 14, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_VERSIONED_OP(Reciprocal, 6, 12, DataTypeImpl::AllIEEEFloatTensorTypes())            \
  NUPHAR_OP(Reciprocal, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                         \
  NUPHAR_VERSIONED_OP(ReduceL1, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())              \
  NUPHAR_OP(ReduceL1, 13, DataTypeImpl::AllFixedSizeTensorTypes())                           \
  NUPHAR_VERSIONED_OP(ReduceL2, 1, 12, DataTypeImpl::AllIEEEFloatTensorTypes())              \
  NUPHAR_OP(ReduceL2, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                           \
  NUPHAR_VERSIONED_OP(ReduceLogSum, 1, 12, DataTypeImpl::AllIEEEFloatTensorTypes())          \
  NUPHAR_OP(ReduceLogSum, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                       \
  NUPHAR_VERSIONED_OP(ReduceLogSumExp, 1, 12, DataTypeImpl::AllIEEEFloatTensorTypes())       \
  NUPHAR_OP(ReduceLogSumExp, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                    \
  NUPHAR_VERSIONED_OP(ReduceMax, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())             \
  NUPHAR_OP(ReduceMax, 13, DataTypeImpl::AllFixedSizeTensorTypes())                          \
  NUPHAR_VERSIONED_OP(ReduceMean, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())            \
  NUPHAR_OP(ReduceMean, 13, DataTypeImpl::AllFixedSizeTensorTypes())                         \
  NUPHAR_VERSIONED_OP(ReduceMin, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())             \
  NUPHAR_OP(ReduceMin, 13, DataTypeImpl::AllFixedSizeTensorTypes())                          \
  NUPHAR_VERSIONED_OP(ReduceProd, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())            \
  NUPHAR_OP(ReduceProd, 13, DataTypeImpl::AllFixedSizeTensorTypes())                         \
  NUPHAR_VERSIONED_OP(ReduceSum, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())             \
  NUPHAR_OP(ReduceSum, 13, DataTypeImpl::AllFixedSizeTensorTypes())                          \
  NUPHAR_VERSIONED_OP(ReduceSumSquare, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())       \
  NUPHAR_OP(ReduceSumSquare, 13, DataTypeImpl::AllFixedSizeTensorTypes())                    \
  NUPHAR_VERSIONED_OP(Reshape, 5, 13, DataTypeImpl::AllFixedSizeTensorTypes())               \
  NUPHAR_OP(Reshape, 14, DataTypeImpl::AllFixedSizeTensorTypes())                            \
  NUPHAR_OP(ScaledTanh, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                          \
  NUPHAR_OP(Selu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Shape, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())                 \
  NUPHAR_OP(Shape, 13, DataTypeImpl::AllFixedSizeTensorTypes())                              \
  NUPHAR_VERSIONED_OP(Sigmoid, 6, 12, {DataTypeImpl::GetTensorType<float>()})                \
  NUPHAR_OP(Sigmoid, 13, {DataTypeImpl::GetTensorType<float>()})                             \
  NUPHAR_VERSIONED_OP(Slice, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())                 \
  NUPHAR_OP(Slice, 13, DataTypeImpl::AllFixedSizeTensorTypes())                              \
  NUPHAR_VERSIONED_OP(Softmax, 1, 12, DataTypeImpl::AllIEEEFloatTensorTypes())               \
  NUPHAR_OP(Softplus, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                            \
  NUPHAR_OP(Softsign, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                            \
  NUPHAR_VERSIONED_OP(Split, 2, 12, DataTypeImpl::AllIEEEFloatTensorTypes())                 \
  NUPHAR_OP(Split, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                              \
  NUPHAR_VERSIONED_OP(Squeeze, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())               \
  NUPHAR_OP(Squeeze, 13, DataTypeImpl::AllFixedSizeTensorTypes())                            \
  NUPHAR_VERSIONED_OP(Sqrt, 6, 12, DataTypeImpl::AllIEEEFloatTensorTypes())                  \
  NUPHAR_OP(Sqrt, 13, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_VERSIONED_OP(Sub, 7, 13, DataTypeImpl::AllFixedSizeTensorTypes())                   \
  NUPHAR_OP(Sub, 14, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Sum, 6, 12, DataTypeImpl::AllFixedSizeTensorTypes())                   \
  NUPHAR_OP(Sum, 13, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_VERSIONED_OP(Tanh, 6, 12, {DataTypeImpl::GetTensorType<float>()})                   \
  NUPHAR_OP(Tanh, 13, {DataTypeImpl::GetTensorType<float>()})                                \
  NUPHAR_OP(ThresholdedRelu, 1, DataTypeImpl::AllFixedSizeTensorTypes())                     \
  NUPHAR_VERSIONED_OP(Tile, 6, 12, DataTypeImpl::AllFixedSizeTensorTypes())                  \
  NUPHAR_OP(Tile, 13, DataTypeImpl::AllFixedSizeTensorTypes())                               \
  NUPHAR_VERSIONED_OP(Transpose, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())             \
  NUPHAR_OP(Transpose, 13, DataTypeImpl::AllFixedSizeTensorTypes())                          \
  NUPHAR_VERSIONED_OP(Unsqueeze, 1, 12, DataTypeImpl::AllFixedSizeTensorTypes())             \
  NUPHAR_OP(Unsqueeze, 13, DataTypeImpl::AllFixedSizeTensorTypes())                          \
  NUPHAR_OP(Where, 9, DataTypeImpl::AllFixedSizeTensorTypes())

}  // namespace nuphar
}  // namespace onnxruntime

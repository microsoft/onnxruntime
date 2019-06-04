// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"
#include "core/providers/nuphar/compiler/tvm_compiler.h"
#include "core/providers/nuphar/runtime/compute_ctx.h"
#include "core/providers/nuphar/runtime/exec_block.h"
#include "core/codegen/target/tvm_context.h"
#include "core/providers/nuphar/compiler/tvm_initializer.h"

#include "core/graph/graph.h"

#include <tvm/build_module.h>

namespace onnxruntime {

class NupharExecutionProvider;

namespace nuphar {

using TryGetConstantFunc = std::function<bool(const std::string&, const Tensor** tensor)>;

class NupharFunctionState;
using NupharFuncStateToComputeCtxMap =
    std::unordered_map<const NupharFunctionState*, std::unique_ptr<tvm_codegen::NupharComputeCtx>>;

class NupharFunctionState {
 public:
  explicit NupharFunctionState(
      const Node& fused_node,
      TryGetConstantFunc try_get_constant_func,  // TODO: remove this
      const ComputeContext& ctx,
      const NupharExecutionProvider* provider);

  explicit NupharFunctionState(
      const OpKernelInfo& info);

  ~NupharFunctionState();

  Status Compute(OpKernelContext* op_kernel_context) const;

 private:
  void Init(const Node& node, TryGetConstantFunc try_get_constant_func);

  const NupharExecutionProvider* provider_;

  std::unique_ptr<tvm_codegen::TVMCompiler> tvm_compiler_;

  // new holder for iniitializer
  tvm_codegen::InitializerMap initializer_map_;

  Status codegen_status_;

  // Hold NupharFuncInfo from codegen.
  std::unique_ptr<tvm_codegen::NupharFuncInfo> func_info_;

  // ExecBlocks of runtime
  std::vector<std::unique_ptr<tvm_codegen::ExecBlock>> exec_blocks_;

  // TODO: confirm whether it is useful
  ComputeContext ctx_;  // the compute context from IExecutionProvider::Compile interface

  static thread_local std::unique_ptr<NupharFuncStateToComputeCtxMap> nuphar_compute_ctx_map_;
};

#define DISABLE_MACRO(X)

#define LIST_NUPHAR_OPS()                                                                    \
  NUPHAR_OP(Abs, 6, DataTypeImpl::AllFixedSizeTensorTypes())                                 \
  NUPHAR_OP(Add, 7, DataTypeImpl::AllFixedSizeTensorTypes())                                 \
  NUPHAR_OP(ArgMax, 1, DataTypeImpl::AllFixedSizeTensorTypes())                              \
  NUPHAR_OP(ArgMin, 1, DataTypeImpl::AllFixedSizeTensorTypes())                              \
  DISABLE_MACRO(NUPHAR_OP(AveragePool, 7, DataTypeImpl::AllFixedSizeTensorTypes()))          \
  NUPHAR_OP(Ceil, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_OP(Clip, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_OP(Concat, 4, DataTypeImpl::AllFixedSizeTensorTypes())                              \
  DISABLE_MACRO(NUPHAR_OP(Conv, 1, DataTypeImpl::AllIEEEFloatTensorTypes()))                 \
  NUPHAR_OP(Crop, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_OP(Div, 7, DataTypeImpl::AllFixedSizeTensorTypes())                                 \
  NUPHAR_OP(Dropout, 7, DataTypeImpl::AllFixedSizeTensorTypes())                             \
  NUPHAR_OP(Elu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                 \
  NUPHAR_OP(Equal, 7, DataTypeImpl::AllFixedSizeTensorTypes())                               \
  NUPHAR_OP(Exp, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                 \
  NUPHAR_VERSIONED_OP(Flatten, 1, 8, DataTypeImpl::AllIEEEFloatTensorTypes())                \
  NUPHAR_OP(Flatten, 9, DataTypeImpl::AllIEEEFloatTensorTypes())                             \
  NUPHAR_OP(Floor, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_VERSIONED_OP(Gemm, 7, 8, DataTypeImpl::AllIEEEFloatTensorTypes())                   \
  NUPHAR_OP(Gemm, 9, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  DISABLE_MACRO(NUPHAR_OP(GlobalAveragePool, 1, DataTypeImpl::AllFixedSizeTensorTypes()))    \
  DISABLE_MACRO(NUPHAR_OP(GlobalMaxPool, 1, DataTypeImpl::AllFixedSizeTensorTypes()))        \
  NUPHAR_OP(Greater, 9, DataTypeImpl::AllFixedSizeTensorTypes())                             \
  NUPHAR_OP(HardSigmoid, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                         \
  NUPHAR_OP(Identity, 1, DataTypeImpl::AllFixedSizeTensorTypes())                            \
  NUPHAR_OP(LeakyRelu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                           \
  NUPHAR_OP(Less, 9, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_OP(Log, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                 \
  NUPHAR_OP(LogSoftmax, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                          \
  DISABLE_MACRO(NUPHAR_OP(LSTM, 7, DataTypeImpl::AllIEEEFloatTensorTypes()))                 \
  NUPHAR_VERSIONED_OP(MatMul, 1, 8, DataTypeImpl::AllIEEEFloatTensorTypes())                 \
  NUPHAR_OP(MatMul, 9, DataTypeImpl::AllIEEEFloatTensorTypes())                              \
  NUPHAR_OP(Max, 8, DataTypeImpl::AllFixedSizeTensorTypes())                                 \
  DISABLE_MACRO(NUPHAR_VERSIONED_OP(MaxPool, 1, 7, DataTypeImpl::AllFixedSizeTensorTypes())) \
  DISABLE_MACRO(NUPHAR_OP(MaxPool, 8, DataTypeImpl::AllFixedSizeTensorTypes()))              \
  NUPHAR_OP(Min, 8, DataTypeImpl::AllFixedSizeTensorTypes())                                 \
  NUPHAR_OP(Mul, 7, DataTypeImpl::AllFixedSizeTensorTypes())                                 \
  NUPHAR_OP(Neg, 6, DataTypeImpl::AllFixedSizeTensorTypes())                                 \
  NUPHAR_OP(Pad, 2, DataTypeImpl::AllIEEEFloatTensorTypes())                                 \
  NUPHAR_OP(ParametricSoftplus, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                  \
  NUPHAR_OP(PRelu, 7, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_OP(Relu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_OP(Reciprocal, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                          \
  NUPHAR_OP(ReduceL1, 1, DataTypeImpl::AllFixedSizeTensorTypes())                            \
  NUPHAR_OP(ReduceL2, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                            \
  NUPHAR_OP(ReduceLogSum, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                        \
  NUPHAR_OP(ReduceLogSumExp, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                     \
  NUPHAR_OP(ReduceMax, 1, DataTypeImpl::AllFixedSizeTensorTypes())                           \
  NUPHAR_OP(ReduceMean, 1, DataTypeImpl::AllFixedSizeTensorTypes())                          \
  NUPHAR_OP(ReduceMin, 1, DataTypeImpl::AllFixedSizeTensorTypes())                           \
  NUPHAR_OP(ReduceProd, 1, DataTypeImpl::AllFixedSizeTensorTypes())                          \
  NUPHAR_OP(ReduceSum, 1, DataTypeImpl::AllFixedSizeTensorTypes())                           \
  NUPHAR_OP(ReduceSumSquare, 1, DataTypeImpl::AllFixedSizeTensorTypes())                     \
  NUPHAR_OP(Reshape, 5, DataTypeImpl::AllFixedSizeTensorTypes())                             \
  NUPHAR_OP(ScaledTanh, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                          \
  NUPHAR_OP(Selu, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_OP(Sigmoid, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                             \
  NUPHAR_VERSIONED_OP(Slice, 1, 9, DataTypeImpl::AllFixedSizeTensorTypes())                  \
  NUPHAR_OP(Softmax, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                             \
  NUPHAR_OP(Softplus, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                            \
  NUPHAR_OP(Softsign, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                            \
  NUPHAR_OP(Split, 2, DataTypeImpl::AllIEEEFloatTensorTypes())                               \
  NUPHAR_OP(Squeeze, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                             \
  NUPHAR_OP(Sqrt, 6, DataTypeImpl::AllIEEEFloatTensorTypes())                                \
  NUPHAR_OP(Sub, 7, DataTypeImpl::AllFixedSizeTensorTypes())                                 \
  NUPHAR_OP(Sum, 8, DataTypeImpl::AllFixedSizeTensorTypes())                                 \
  NUPHAR_OP(Tanh, 6, DataTypeImpl::AllFixedSizeTensorTypes())                                \
  NUPHAR_OP(ThresholdedRelu, 1, DataTypeImpl::AllFixedSizeTensorTypes())                     \
  NUPHAR_OP(Transpose, 1, DataTypeImpl::AllFixedSizeTensorTypes())                           \
  NUPHAR_OP(Unsqueeze, 1, DataTypeImpl::AllIEEEFloatTensorTypes())                           \
  NUPHAR_OP(Where, 9, DataTypeImpl::AllFixedSizeTensorTypes())

}  // namespace nuphar
}  // namespace onnxruntime

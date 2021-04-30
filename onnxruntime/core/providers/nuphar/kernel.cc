// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/kernel.h"

#include "core/codegen/passes/utils/codegen_context.h"
#include "core/codegen/common/profile.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/nuphar/common/analysis/subgraph_codegen_stats.h"
#include "core/providers/nuphar/compiler/initializer_info.h"
#include "core/providers/nuphar/nuphar_execution_provider.h"
#include "core/providers/nuphar/partition/subgraph_partitioner.h"
#include "core/providers/nuphar/runtime/sequential/basic.h"
#include "core/providers/nuphar/runtime/sequential/loop.h"

namespace onnxruntime {
namespace nuphar {

thread_local std::unique_ptr<NupharFuncStateToComputeCtxMap> NupharKernelState::nuphar_compute_ctx_map_;

NupharKernelState::NupharKernelState(
    const Node& node,
    const ComputeContext& ctx,
    const NupharExecutionProvider& provider)
    : provider_(provider),
      ctx_(ctx) {
  partition_info_ = std::make_unique<OrtSubgraphAllocationInfo>(node);

  std::vector<NupharSubgraphUnit> subgraphs;

  // create a partitioner
  SubgraphPartitioner subgraph_partitioner;
  subgraph_partitioner.Partition(
      node,
      subgraphs,
      [&](const std::string& name) { return provider_.GetConstantInitializer(name); });

  for (auto& subgraph : subgraphs) {
    Compile(subgraph);
    if (!codegen_status_.IsOK()) {
      return;  // early return
    }
  }

  // Currently BuildExecBlocksAndCalls is inserted here
  // TODO: after AOT support, we should move it to a proper location
  BuildExecBlocksAndCalls(subgraphs);
}

void NupharKernelState::Compile(const NupharSubgraphUnit& subgraph) {
  // TODO: rename tvm_target to a proper name
  auto tvm_target = provider_.GetTVMTarget();

  NupharCompiler tvm_compiler(subgraph,
                              generated_initializers_,
                              provider_.GetNupharCodeGenHandle());

  codegen_status_ = tvm_compiler.Build(subgraph);

  if (codegen_status_.IsOK()) {
    func_infos_.emplace_back(std::make_unique<NupharFuncInfo>());
    codegen_status_ = tvm_compiler.Lower(subgraph,
                                         tvm_target,
                                         provider_.GetTVMHostTarget(),
                                         func_infos_.back().get(),
                                         partition_info_.get());
  }
}

void NupharKernelState::BuildExecBlocksAndCalls(const std::vector<NupharSubgraphUnit>& subgraphs) {
  // create ExecBlocks
  for (size_t idx = 0; idx < subgraphs.size(); ++idx) {
    CreateExecBlock(exec_blocks_,
                    func_infos_[idx].get(),
                    subgraphs[idx],
                    provider_.GetNupharRuntimeHandle()->enable_model_parallelism);
  }

  // create calls
  for (const auto& eb : exec_blocks_) {
    exec_block_calls_.push_back(eb.get());
  }
}

NupharKernelState::~NupharKernelState() {
  if (nullptr != nuphar_compute_ctx_map_)
    nuphar_compute_ctx_map_->erase(this);
}

Status NupharKernelState::Compute(OpKernelContext* op_kernel_context) const {
  if (!codegen_status_.IsOK()) {
    return codegen_status_;
  }

  // Create the unordered_map if it not exist
  if (nullptr == nuphar_compute_ctx_map_) {
    nuphar_compute_ctx_map_ = std::make_unique<NupharFuncStateToComputeCtxMap>();
  }

  // Create KernelComputeCtx if it not exist
  if (nuphar_compute_ctx_map_->find(this) == nuphar_compute_ctx_map_->end()) {
    std::function<void*(size_t)> data_alloc_func =
        [this](size_t bytes) { return provider_.GetNupharRuntimeHandle()->allocator->Alloc(bytes); };

    nuphar_compute_ctx_map_->emplace(
        std::make_pair(this,
                       std::make_unique<KernelComputeCtx>(
                           provider_.GetNupharRuntimeHandle(),
                           provider_.GetTLSRealizedDims(),
                           data_alloc_func,
                           partition_info_->offset_count)));
  }

  KernelComputeCtx* compute_ctx = nuphar_compute_ctx_map_->find(this)->second.get();

  ORT_ENFORCE_DEBUG(nullptr != compute_ctx);

  compute_ctx->Bind(op_kernel_context);

  for (auto* call : exec_block_calls_) {
    CODEGEN_PROFILER_EVENT(call->Name());
    call->Run(compute_ctx);
  }

  return Status::OK();
}

// dummy kernel for single node, for registration purpose only
class NupharKernel : public OpKernel {
 public:
  explicit NupharKernel(const OpKernelInfo& info)
      : OpKernel(info) {
    ORT_ENFORCE(false);  // not supposed to instantiate
  }

  Status Compute(OpKernelContext* context) const override {
    return Status::OK();
  }
};

}  // namespace nuphar

#define NUPHAR_OP(name, ver, types)                  \
  ONNX_OPERATOR_KERNEL_EX(                           \
      name,                                          \
      kOnnxDomain,                                   \
      ver,                                           \
      kNupharExecutionProvider,                      \
      KernelDefBuilder().TypeConstraint("T", types), \
      nuphar::NupharKernel);

#define NUPHAR_VERSIONED_OP(name, start_ver, end_ver, types) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                         \
      name,                                                  \
      kOnnxDomain,                                           \
      start_ver,                                             \
      end_ver,                                               \
      kNupharExecutionProvider,                              \
      KernelDefBuilder().TypeConstraint("T", types),         \
      nuphar::NupharKernel);

LIST_NUPHAR_OPS()

#undef NUPHAR_OP

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast,
    kOnnxDomain,
    6,
    8,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T2", DataTypeImpl::AllFixedSizeTensorExceptHalfTypes()),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    Cast,
    kOnnxDomain,
    9,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::AllFixedSizeTensorExceptHalfTypes())
        .TypeConstraint("T2", DataTypeImpl::AllFixedSizeTensorExceptHalfTypes()),
    nuphar::NupharKernel);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gather,
    kOnnxDomain,
    1,
    10,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                        DataTypeImpl::GetTensorType<int64_t>()}),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kOnnxDomain,
    11,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                        DataTypeImpl::GetTensorType<int64_t>()}),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    GatherElements,
    kOnnxDomain,
    11,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                        DataTypeImpl::GetTensorType<int64_t>()}),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    MatMulInteger,
    kOnnxDomain,
    10,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<int8_t>(),
                               DataTypeImpl::GetTensorType<uint8_t>()})
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int8_t>(),
                               DataTypeImpl::GetTensorType<uint8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    MatMulInteger16,
    kMSDomain,
    1,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<int16_t>()})
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<int16_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int32_t>()),
    nuphar::NupharKernel);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Scan,
    kOnnxDomain,
    9,
    10,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    Scan,
    kOnnxDomain,
    11,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
    nuphar::NupharKernel);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Scatter,
    kOnnxDomain,
    9,
    10,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                        DataTypeImpl::GetTensorType<int64_t>()}),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    Scatter,
    kOnnxDomain,
    11,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                        DataTypeImpl::GetTensorType<int64_t>()}),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    ScatterElements,
    kOnnxDomain,
    11,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                        DataTypeImpl::GetTensorType<int64_t>()}),
    nuphar::NupharKernel);

}  // namespace onnxruntime

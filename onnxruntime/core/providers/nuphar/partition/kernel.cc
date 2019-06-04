// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/nuphar/partition/kernel.h"
#include "core/providers/nuphar/nuphar_execution_provider.h"
#include "core/providers/nuphar/common/analysis/subgraph_codegen_stats.h"
#include "core/providers/nuphar/compiler/tvm_initializer.h"
#include "core/codegen/target/tvm_context.h"

#include "core/providers/nuphar/partition/subgraph_partitioner.h"

#include "core/providers/nuphar/runtime/sequential/basic.h"
#include "core/providers/nuphar/runtime/sequential/loop.h"

namespace onnxruntime {
namespace nuphar {

thread_local std::unique_ptr<NupharFuncStateToComputeCtxMap> NupharKernelState::nuphar_compute_ctx_map_;

NupharKernelState::NupharKernelState(
    const Node& node,
    TryGetConstantFunc try_get_constant_func,
    const ComputeContext& ctx,
    const NupharExecutionProvider* provider)
    : provider_(provider),
      ctx_(ctx) {
  // create initializers
  node.ForEachDef(
      [&](const NodeArg& def, bool is_input) {
        if (!is_input)
          return;

        const Tensor* tensor = nullptr;
        if (try_get_constant_func(def.Name(), &tensor)) {
          initializers_.emplace(def.Name(), tensor);
        }
      });

  partition_info_ = std::make_unique<OrtSubgraphAllocationInfo>(node);

  std::vector<NupharSubgraphUnit> subgraphs;

  // create a partitioner
  SubgraphPartitioner subgraph_partitioner;
  subgraph_partitioner.Partition(node, subgraphs, initializers_);

  for (auto& subgraph : subgraphs) {
    Compile(subgraph);
    if (!codegen_status_.IsOK()) {
      break;  // early return
    }
  }
}

// for Single node
NupharKernelState::NupharKernelState(
    const OpKernelInfo& info)
    : provider_(dynamic_cast<const NupharExecutionProvider*>(info.GetExecutionProvider())) {
  Compile(info.node(),
          [&](const std::string& name, const Tensor** tensor) {
            return info.TryGetConstantInput(name, tensor);
          });
}

void NupharKernelState::Compile(const NupharSubgraphUnit& subgraph) {
  // TODO: rename tvm_target to a proper name
  auto tvm_target = provider_->GetTVMTarget();

  tvm_codegen::NupharCompiler tvm_compiler(subgraph,
                                           generated_initailizers_,
                                           provider_->GetNupharCodeGenHandle());

  codegen_status_ = tvm_compiler.Build(subgraph);

  if (codegen_status_.IsOK()) {
    func_infos_.push_back(std::move(std::make_unique<tvm_codegen::NupharFuncInfo>()));
    codegen_status_ = tvm_compiler.Lower(subgraph,
                                         tvm_target,
                                         provider_->GetTVMHostTarget(),
                                         func_infos_.back().get(),
                                         partition_info_.get());
  }
  if (codegen_status_.IsOK()) {
    if (subgraph.IsSingleNode() && subgraph.nodes.front()->OpType() == "Scan") {
      exec_blocks_.push_back(
          std::move(std::make_unique<tvm_codegen::LoopExecBlock>(func_infos_.back().get(), "nuphar_exec_" + subgraph.Name())));
    } else {
      exec_blocks_.push_back(
          std::move(std::make_unique<tvm_codegen::BasicExecBlock>(func_infos_.back().get(), "nuphar_exec_" + subgraph.Name())));
    }
  }
}

// single node
void NupharKernelState::Compile(
    const Node& node,
    TryGetConstantFunc try_get_constant_func) {
  // TODO: rename tvm_target to a proper name
  auto tvm_target = provider_->GetTVMTarget();

  node.ForEachDef(
      [&](const NodeArg& def, bool is_input) {
        if (!is_input)
          return;

        const Tensor* tensor = nullptr;
        if (try_get_constant_func(def.Name(), &tensor)) {
          initializers_.emplace(def.Name(), tensor);
        }
      });

  tvm_codegen::NupharCompiler tvm_compiler(node,
                                           initializers_,
                                           generated_initailizers_,
                                           provider_->GetNupharCodeGenHandle());

  codegen_status_ = tvm_compiler.Build(node);

  if (codegen_status_.IsOK()) {
    func_infos_.push_back(std::move(std::make_unique<tvm_codegen::NupharFuncInfo>()));
    codegen_status_ = tvm_compiler.Lower(node,
                                         tvm_target,
                                         provider_->GetTVMHostTarget(),
                                         func_infos_.back().get());
  }

  // TODO: move this to another function for code readabilty
  if (node.OpType() == "Scan") {
    exec_blocks_.push_back(
        std::move(std::make_unique<tvm_codegen::LoopExecBlock>(func_infos_.back().get(), "nuphar_exec_" + node.Name())));
  } else {
    exec_blocks_.push_back(
        std::move(std::make_unique<tvm_codegen::BasicExecBlock>(func_infos_.back().get(), "nuphar_exec_" + node.Name())));
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
        [this](size_t bytes) { return provider_->GetNupharRuntimeHandle()->allocator->Alloc(bytes); };

    nuphar_compute_ctx_map_->emplace(
        std::make_pair(this,
                       std::make_unique<tvm_codegen::KernelComputeCtx>(
                           provider_->GetNupharRuntimeHandle(),
                           provider_->GetTLSRealizedDims(),
                           data_alloc_func,
                           partition_info_->offset_count)));
  }

  tvm_codegen::KernelComputeCtx* compute_ctx = nuphar_compute_ctx_map_->find(this)->second.get();

  ORT_ENFORCE_DEBUG(nullptr != compute_ctx);

  compute_ctx->Bind(op_kernel_context);

  for (const auto& exec : exec_blocks_) {
    exec->Run(compute_ctx);
  }

  return Status::OK();
}

// This is mainly for single node
class NupharKernel : public OpKernel {
 public:
  explicit NupharKernel(const OpKernelInfo& info)
      : OpKernel(info),
        func_state_(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    return func_state_.Compute(context);
  }

 private:
  NupharKernelState func_state_;
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

// ops that have multiple type constraints

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Cast,
    kOnnxDomain,
    6,
    8,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T2", DataTypeImpl::AllFixedSizeTensorTypes()),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    Cast,
    kOnnxDomain,
    9,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T2", DataTypeImpl::AllFixedSizeTensorTypes()),
    nuphar::NupharKernel);

ONNX_OPERATOR_KERNEL_EX(
    Gather,
    kOnnxDomain,
    1,
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
    Scan,
    kOnnxDomain,
    9,
    kNupharExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
    nuphar::NupharKernel);

}  // namespace onnxruntime

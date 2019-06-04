// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/build_module.h>

#include "nuphar_allocator.h"
#include "core/providers/nuphar/common/analysis/graph_stats.h"
#include "core/codegen/common/common.h"
#include "core/providers/nuphar/compiler/traverse_shape_infer.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"

#include "core/providers/nuphar/compiler/tvm_manager.h"

#include "core/providers/nuphar/runtime/handle.h"

namespace onnxruntime {

// Forward declaration
class CodeGenTarget;

// By default, construct either "llvm" or "stackvm" TVM target, for which the default device_type is kDLCPU.
#ifdef USE_TVM_WITH_LLVM
constexpr const char* default_nuphar_target_str = "llvm";
#else
constexpr const char* default_nuphar_target_str = "stackvm";
#endif  // USE_TVM_WITH_LLVM

// Information needed to construct Nuphar execution providers.
struct NupharExecutionProviderInfo {
  int device_id{0};
  // By default, let provider decide the target by passing in empty string.
  std::string target_str;
  bool enable_per_node_parallel;  // TODO: remove

  // this flag set TVM build_config with data_alignment=1, at the cost of performance
  bool allow_unaligned_buffers;

  explicit NupharExecutionProviderInfo(bool unaligned_buffers,
                                       int dev_id = 0,
                                       const std::string& tgt_str = "",
                                       bool per_node_parallel = true)
      : device_id(dev_id),
        target_str(tgt_str),
        enable_per_node_parallel(per_node_parallel),
        allow_unaligned_buffers(unaligned_buffers) {}
  NupharExecutionProviderInfo() = default;
};

class NupharExecutionProvider : public IExecutionProvider {
 public:
  explicit NupharExecutionProvider(const NupharExecutionProviderInfo& info);

  virtual ~NupharExecutionProvider() = default;

  Status CopyTensor(const Tensor& src, Tensor& dst) const override;

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& kernel_registries) const override;

  Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override;

  const void* GetExecutionHandle() const noexcept override {
    // The Nuphar interface does not return anything interesting.
    return nullptr;
  }

  Status OnRunStart() override {
    if (tls_realized_dims_ != nullptr) {
      // at frame start, reset realized_dims since new execution frame may have different dynamic value
      for (auto& pair : *(tls_realized_dims_.get())) {
        pair.second = Dimension_Unknown;
      }
    } else {
      tls_realized_dims_ = std::make_unique<std::unordered_map<std::string, int64_t>>();
    }
    return Status::OK();
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

  const TVMContext& GetTVMContext() const {
    return tvm_ctx_;
  }

  tvm::Target GetTVMTarget() const {
    return tvm_target_;
  }

  tvm::Target GetTVMHostTarget() const {
    return tvm_target_;
  }

  // TODO remove
  const std::shared_ptr<ShapeExprContext>& GetShapeInfernece() const {
    return whole_graph_shape_infer_;
  }

  // NOTE: realized_dims_ is thread_local, so it can only be accessed in execution thread (not ctor/dtor)
  std::unordered_map<std::string, int64_t>& GetTLSRealizedDims() const {
    return *(tls_realized_dims_.get());
  }

  // TODO: refactor after adding multi-target
  // TODO: rename
  const tvm_codegen::NupharCodeGenHandle* GetNupharCodeGenHandle() const {
    ORT_ENFORCE(codegen_handles_.size() > 0);
    return codegen_handles_.front().get();
  }

  const nuphar::NupharRuntimeHandle* GetNupharRuntimeHandle() const {
    return runtime_handle_.get();
  }

  const int GetDomainVersion(const std::string& name) const {
    ORT_ENFORCE(domain_versions_.count(name));
    return domain_versions_[name];
  }

 private:
  void CreateTVMTarget();

  Status SaveInitializer(
      const std::string& name,
      const ONNX_NAMESPACE::TensorProto* proto) const;

 private:
  // TODO remove
  std::unique_ptr<CodeGenTarget> codegen_target_;

  // TODO: move all tvm related code a manager
  tvm::Target tvm_target_;
  tvm::Target tvm_host_target_;
  TVMContext tvm_ctx_;

  // shape inference
  std::shared_ptr<ShapeExprContext> whole_graph_shape_infer_;

  // graph stats
  std::unique_ptr<codegen::OrtGraphStats> graph_stats_;

  // mapping from symbolic dimension to actual value
  static thread_local std::unique_ptr<std::unordered_map<std::string, int64_t>> tls_realized_dims_;

  std::unique_ptr<tvm_codegen::TVMCodeGenManager> tvm_codegen_manager_;

  // codegen_handles_ holds a list of NupharCodeGenHandle .
  // Why a list? it is for multi-target support later
  // The current release supports one codegen target.
  // TODO: support multi-target support
  std::vector<std::unique_ptr<tvm_codegen::NupharCodeGenHandle>> codegen_handles_;

  std::unique_ptr<nuphar::NupharRuntimeHandle> runtime_handle_;

  mutable std::shared_ptr<KernelRegistry> kernel_registry_;

  // a copy of Node to keep Node's life-time
  // TODO: remove this after completely decoupling runtime and compiler
  std::vector<onnxruntime::Node*> compiled_nodes_;

  mutable std::unordered_map<std::string, std::unique_ptr<Tensor>> initializers_used_in_compiled_nodes_;
  mutable std::unordered_map<std::string, int> domain_versions_;
};

}  // namespace onnxruntime

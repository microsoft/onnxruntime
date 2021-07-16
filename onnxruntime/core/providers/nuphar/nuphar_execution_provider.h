// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/providers/nuphar/common/analysis/graph_stats.h"
#include "core/providers/nuphar/compiler/codegen_manager.h"
#include "core/providers/nuphar/compiler/traverse_shape_infer.h"
#include "core/providers/nuphar/runtime/handle.h"

#include <tvm/build_module.h>

namespace onnxruntime {

// Forward declaration
class CodeGenTarget;

// By default, construct either "llvm" or "stackvm" TVM target, for which the default device_type is kDLCPU.
constexpr const char* llvm_target_str = "llvm";
constexpr const char* stackvm_target_str = "stackvm";

#ifdef USE_TVM_WITH_LLVM
constexpr const char* default_nuphar_target_str = llvm_target_str;
#else
constexpr const char* default_nuphar_target_str = stackvm_target_str;
#endif  // USE_TVM_WITH_LLVM

// Information needed to construct Nuphar execution providers.
struct NupharExecutionProviderInfo {
  // this flag set TVM build_config with data_alignment=1, at the cost of performance
  bool allow_unaligned_buffers;

  // this string contains key/value pairs like:
  // key1:value1, key2:value2, ...
  // it would override environment variables for settings
  std::string settings;

  explicit NupharExecutionProviderInfo(bool unaligned_buffers,
                                       const std::string& str_settings = "")
      : allow_unaligned_buffers(unaligned_buffers),
        settings(str_settings) {}
  NupharExecutionProviderInfo() = default;
};

class NupharExecutionProvider : public IExecutionProvider {
 public:
  explicit NupharExecutionProvider(const NupharExecutionProviderInfo& info);

  virtual ~NupharExecutionProvider() = default;

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

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    // do not register individual kernels
    return std::make_shared<KernelRegistry>();
  }

  // internal registry for checking if op is supported
  std::shared_ptr<KernelRegistry> GetKernelRegistryInternal() const;

  tvm::Target GetTVMTarget() const {
    return tvm_target_;
  }

  tvm::Target GetTVMHostTarget() const {
    return tvm_host_target_;
  }

  // NOTE: realized_dims_ is thread_local, so it can only be accessed in execution thread (not ctor/dtor)
  std::unordered_map<std::string, int64_t>& GetTLSRealizedDims() const {
    return *(tls_realized_dims_.get());
  }

  const nuphar::NupharCodeGenHandle* GetNupharCodeGenHandle() const {
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

  const Tensor* GetConstantInitializer(const std::string& name) const {
    auto iter = constant_initializers_used_in_compiled_nodes_.find(name);
    if (iter == constant_initializers_used_in_compiled_nodes_.end())
      return nullptr;
    return iter->second.get();
  }

 private:
  void CreateTVMTarget();

  Status SaveInitializer(
      const std::string& name,
      const ONNX_NAMESPACE::TensorProto* proto) const;

 private:
  // TODO move this to another place
  std::unique_ptr<CodeGenTarget> codegen_target_;
  // TODO: move all tvm related code a manager
  tvm::Target tvm_target_;
  tvm::Target tvm_host_target_;
  TVMContext tvm_ctx_;

  // shape inference
  std::shared_ptr<nuphar::ShapeExprContext> whole_graph_shape_infer_;

  // mapping from symbolic dimension to actual value
  static thread_local std::unique_ptr<std::unordered_map<std::string, int64_t>> tls_realized_dims_;

  std::unique_ptr<nuphar::TVMCodeGenManager> tvm_codegen_manager_;

  // codegen_handles_ holds a list of NupharCodeGenHandle .
  // Why a list? it is for multi-target support
  // The current release supports one codegen target.
  // TODO: support multi-target support
  std::vector<std::unique_ptr<nuphar::NupharCodeGenHandle>> codegen_handles_;

  std::unique_ptr<nuphar::NupharRuntimeHandle> runtime_handle_;

  mutable std::shared_ptr<KernelRegistry> kernel_registry_;

  mutable std::unordered_map<std::string, std::unique_ptr<Tensor>> constant_initializers_used_in_compiled_nodes_;
  mutable std::unordered_map<std::string, int> domain_versions_;

  // used to create unique fused node name, make it thread_local because
  // subsession of a model with subgraph may create multiple instances of EPs,
  // and there might be multiple inference sessions running different models concurrently
  static thread_local int per_model_fused_count_;
};

}  // namespace onnxruntime

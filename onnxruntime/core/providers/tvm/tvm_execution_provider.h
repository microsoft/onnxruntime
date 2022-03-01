// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_EXECUTION_PROVIDER_H
#define TVM_EXECUTION_PROVIDER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/platform/ort_mutex.h"

#include "tvm_common.h"
#include "tvm_execution_provider_info.h"

namespace onnxruntime {

namespace tvm {
namespace env_vars {
   static const std::string kDumpSubgraphs = "ORT_TVM_DUMP_SUBGRAPHS";
}  // namespace env_vars
}  // namespace tvm

class TVMRunner;

class TvmExecutionProvider : public IExecutionProvider {
  friend TVMRunner;

  using TVMTensorShape = std::vector<int64_t>;
  using TVMTensorShapes = std::vector<TVMTensorShape>;
  using TVMRunners = std::unordered_map<std::string, std::shared_ptr<TVMRunner>>;
  using TVMModules = std::unordered_map<std::string, std::shared_ptr<TvmModule>>;
 public:
  explicit TvmExecutionProvider(const TvmExecutionProviderInfo& info);
  virtual ~TvmExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

 private:
  bool GPUTargetCheck() const;
  size_t split(const std::string &txt, std::vector<std::string> &strs, char ch) const;
  void ProcessInfo();
  void ProcessCPUTarget();
  void ProcessGPUTarget();
  void PrintProviderOptions() const;
  // Bindings for compute info
  int CreateStateFunc(ComputeContext*, FunctionState*);
  TvmModule* CompileFunc(std::string func_name, const TVMTensorShapes& input_shapes);
 private:
  TVMRunners runners_;
  std::unordered_map<std::string, std::string> buffers_;
  std::unordered_map<std::string, int> opsets_;
  std::unordered_map<std::string, std::string>  model_paths_;
  bool dump_subgraphs_ = false;
  OrtMutex tvm_mu_;
  AllocatorPtr allocator_;
  TvmExecutionProviderInfo info_;
  TVMModules modules_;
};

}  // namespace onnxruntime

#endif  // TVM_EXECUTION_PROVIDER_H

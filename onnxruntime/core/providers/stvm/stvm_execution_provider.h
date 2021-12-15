// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef STVM_EXECUTION_PROVIDER_H
#define STVM_EXECUTION_PROVIDER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "core/common/logging/logging.h"
#include "core/framework/execution_provider.h"
#include "core/providers/stvm/stvm_execution_provider_info.h"
#include "core/platform/ort_mutex.h"

#include "stvm_common.h"

namespace onnxruntime {

namespace stvm_env_vars {
   static const std::string kDumpSubgraphs = "ORT_STVM_DUMP_SUBGRAPHS";
}  // namespace stvm_env_vars

class STVMRunner;

class StvmExecutionProvider : public IExecutionProvider {
  friend STVMRunner;

  using TVMTensorShape = std::vector<int64_t>;
  using TVMTensorShapes = std::vector<TVMTensorShape>;
  using STVMRunners = std::unordered_map<std::string, std::shared_ptr<STVMRunner>>;
  using STVMModules = std::unordered_map<std::string, std::shared_ptr<tvm::runtime::Module>>;
 public:
  explicit StvmExecutionProvider(const StvmExecutionProviderInfo& info);
  virtual ~StvmExecutionProvider();

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
  void PrintInfo() const;
  // Bindings for compute info
  int CreateStateFunc(ComputeContext*, FunctionState*);
  tvm::runtime::Module* CompileFunc(std::string func_name, const TVMTensorShapes& input_shapes);
 private:
  STVMRunners runners_;
  std::unordered_map<std::string, std::string> buffers_;
  std::unordered_map<std::string, int> opsets_;
  std::unordered_map<std::string, std::string>  model_paths_;
  bool dump_subgraphs_ = false;
  OrtMutex stvm_mu_;
  AllocatorPtr allocator_;
  StvmExecutionProviderInfo info_;
  STVMModules modules_;
};

}  // namespace onnxruntime

#endif  // STVM_EXECUTION_PROVIDER_H

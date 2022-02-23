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

#include "tvm_compiler.h"
#include "tvm_runner.h"

namespace onnxruntime {

namespace tvm {
namespace env_vars {
   static const std::string kDumpSubgraphs = "ORT_TVM_DUMP_SUBGRAPHS";
}  // namespace env_vars
}  // namespace tvm

class TvmExecutionProvider : public IExecutionProvider {
  using Compiler = tvm::TVMCompiler;
  using Compilers = std::unordered_map<std::string, std::shared_ptr<Compiler>>;
  using Runner = tvm::TVMRunner;
  using Runners = std::unordered_map<std::string, std::shared_ptr<Runner>>;

  friend Runner;
 public:
  explicit TvmExecutionProvider(const TvmEPOptions& options);
  virtual ~TvmExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

 private:
  int CreateStateFunc(ComputeContext*, FunctionState*);
 private:
  TvmEPOptions options_;
  Compilers compilers_;
  Runners runners_;
  bool dump_subgraphs_ = false;
  OrtMutex tvm_mu_;
  AllocatorPtr allocator_;
};

}  // namespace onnxruntime

#endif  // TVM_EXECUTION_PROVIDER_H

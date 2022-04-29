// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_CORE_PROVIDERS_TVM_TVM_SO_EXECUTION_PROVIDER_H_
#define ONNXRUNTIME_CORE_PROVIDERS_TVM_TVM_SO_EXECUTION_PROVIDER_H_

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
class Graph;
class NodeArg;
namespace tvm {

class TvmSoExecutionProvider : public IExecutionProvider {
  using Compiler = TVMCompilerBase;
  using Compilers = std::unordered_map<std::string, std::shared_ptr<Compiler>>;
  using Runner = TVMRunner;
  using Runners = std::unordered_map<std::string, std::shared_ptr<Runner>>;

 public:
  explicit TvmSoExecutionProvider(const TvmEPOptions& options);
  virtual ~TvmSoExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  common::Status Compile(const std::vector<onnxruntime::Node*>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
  std::unique_ptr<onnxruntime::IDataTransfer> GetDataTransfer() const override;
  AllocatorPtr GetAllocator(int id, OrtMemType mem_type) const override;

 private:
  void printOptions();
  std::shared_ptr<TvmModule> compileModel(const std::string& func_name,
                                          const Graph& graph,
                                          InputsInfoMap& inputs_info);
  void setInputShapesForFreezedNN(const Graph& graph, TVMTensorShapes& input_shapes, InputsInfoMap& all_input_shapes);
  void setInputShapesForUnfreezedNN(const Graph& graph, TVMTensorShapes& input_shapes, InputsInfoMap& all_input_shapes);
  TensorShapeVector getInputShape(const NodeArg* node);
  TensorShapeVector convertTensorShape(const ONNX_NAMESPACE::TensorShapeProto& shape_proto);
  void prepareOutputTensors(std::vector<DLTensor>& output_tensors);
  NodeComputeInfo prepareComputeInfo(const std::string& func_name);
  int createStateFunc(ComputeContext*, FunctionState*);

 private:
  TvmEPOptions options_;
  Compilers compilers_;
  Runners runners_;
  AllocatorPtr allocator_;
};

}  // namespace tvm
}  // namespace onnxruntime

#endif  // ONNXRUNTIME_CORE_PROVIDERS_TVM_TVM_SO_EXECUTION_PROVIDER_H_

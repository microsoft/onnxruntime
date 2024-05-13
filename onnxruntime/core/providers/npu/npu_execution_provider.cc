

#include "core/providers/npu/npu_execution_provider.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"

namespace onnxruntime {
NPUExecutionProvider::NPUExecutionProvider(const NPUExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kNpuExecutionProvider}, info_{info} {}

std::vector<std::unique_ptr<ComputeCapability>>
NPUExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer, const IKernelLookup& /*kernel_lookup*/) const {

  for (const auto& node : graph_viewer.Nodes()) {
    std::cout << " Node Name : " << node.Name() << std::endl;
    std::cout << " Node OpType : " << node.OpType() << std::endl;
    std::cout << " Node Domain : " << node.Domain() << std::endl;
    std::cout << " Input Count : " << node.InputDefs().size() << std::endl;
    std::cout << " Output Count : " << node.OutputDefs().size() << std::endl;
  }

  return {};
}

}  // namespace onnxruntime

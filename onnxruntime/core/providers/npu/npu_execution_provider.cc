

#include "core/providers/npu/npu_execution_provider.h"

namespace onnxruntime {
NPUExecutionProvider::NPUExecutionProvider(const NPUExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kNpuExecutionProvider}, info_{info} {}

std::vector<std::unique_ptr<ComputeCapability>>
GetCapability(const onnxruntime::GraphViewer& graph,
              const IKernelLookup& kernel_lookup) const {
  for (const auto& node : graph.Nodes()) {
    std::cout << " Node Name : " << node.Name() << std::endl;
    std::cout << " Node OpType : " << node.opType() << std::endl;
    std::cout << " Node Domain : " << node.Domain() << std::endl;
    std::cout << " Input Count : " << node.InputDefs().size() << std::endl;
    std::cout << " Output Count : " << node.OutputDefs().size() << std::endl;
  }

  return {};
}

}  // namespace onnxruntime

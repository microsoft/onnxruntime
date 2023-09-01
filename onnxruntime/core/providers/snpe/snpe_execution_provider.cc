// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/snpe/snpe_execution_provider.h"
#include <vector>
#include <memory>
#include <utility>
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/kernel_registry.h"

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  onnxruntime::Status st;
};
}  // namespace

namespace onnxruntime {

constexpr const char* SNPE = "SNPE";

namespace contrib {
namespace snpe {
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kSnpeExecutionProvider, kMSDomain, 1, Snpe);

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterSnpeContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kSnpeExecutionProvider, kMSDomain, 1, Snpe)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

}  // namespace snpe
}  // namespace contrib

KernelRegistryAndStatus GetSnpeKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = ::onnxruntime::contrib::snpe::RegisterSnpeContribKernels(*ret.kernel_registry);
  return ret;
}

std::shared_ptr<KernelRegistry> SNPEExecutionProvider::GetKernelRegistry() const {
  static KernelRegistryAndStatus ret = GetSnpeKernelRegistry();
  // throw if the registry failed to initialize
  ORT_THROW_IF_ERROR(ret.st);
  return ret.kernel_registry;
}

SNPEExecutionProvider::SNPEExecutionProvider(const ProviderOptions& provider_options_map)
    : IExecutionProvider{onnxruntime::kSnpeExecutionProvider}, runtime_options_(provider_options_map) {
}

SNPEExecutionProvider::~SNPEExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
SNPEExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                     const IKernelLookup& kernel_lookup) const {
  std::vector<NodeIndex> candidates;
  for (auto& node_index : graph.GetNodesInTopologicalOrder()) {
    const auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    const auto& node = *p_node;
    if (!node.GetExecutionProviderType().empty()) {
      continue;
    }

    const KernelCreateInfo* snpe_kernel_def = kernel_lookup.LookUpKernel(node);
    if (snpe_kernel_def == nullptr) {
      LOGS_DEFAULT(WARNING) << "Snpe kernel not found in registries for Op type: " << node.OpType()
                            << " node name: " << node.Name();
      continue;
    }

    candidates.push_back(node.Index());
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& node_index : candidates) {
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(node_index);
    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  }
  return result;
}

}  // namespace onnxruntime

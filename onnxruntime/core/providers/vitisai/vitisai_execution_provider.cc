// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vitisai_execution_provider.h"

#include <cassert>
#include <fstream>
#include <istream>

#include "vaip/capability.h"
#include "vaip/global_api.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
constexpr const char* VITISAI = "VITISAI";

VitisAIExecutionProvider::VitisAIExecutionProvider(
    const ProviderOptions& info)
    : IExecutionProvider{onnxruntime::kVitisAIExecutionProvider}, info_(info) {
  CreateKernelRegistry();
}

void VitisAIExecutionProvider::CreateKernelRegistry() {
  for (const auto& domain : get_domains_vitisaiep()) {
    for (const auto* op : domain->custom_ops_) {
      vitisai_optypes_.insert(op->GetName(op));
    }
  }
}

std::shared_ptr<KernelRegistry> VitisAIExecutionProvider::GetKernelRegistry() const { return get_kernel_registry_vitisaiep(); }

std::vector<std::unique_ptr<ComputeCapability>> VitisAIExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph, const IKernelLookup& /*kernel_lookup*/) const {
  if (graph.IsSubgraph()) {
    // VITIS AI EP not support sungraph. Assigned to CPU.
    return {};
  }
  if (execution_providers_) {
    // Only compiling a model once is currently supported
    return {};
  }
  execution_providers_ = std::make_unique<my_ep_t>(compile_onnx_model(graph, *GetLogger(), info_));
  auto result = vaip::GetComputeCapabilityOps(graph, execution_providers_.get(), vitisai_optypes_);
  size_t index = 0u;
  for (auto& ep : **execution_providers_) {
    result.emplace_back(vaip::XirSubgraphToComputeCapability1(graph, ep.get(), index));
    index = index + 1;
  }
  return result;
}

common::Status VitisAIExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                 std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    NodeComputeInfo compute_info;
    auto& attrs = fused_node_graph.fused_node.get().GetAttributes();
    assert(attrs.count("index"));
    size_t index = attrs.at("index").i();
    compute_info.create_state_func = [this, index](ComputeContext* context, FunctionState* state) {
      auto* p = (**this->execution_providers_)[index]->compile().release();
      *state = p;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state) {
        delete reinterpret_cast<vaip_core::CustomOp*>(state);
      }
    };
    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      reinterpret_cast<vaip_core::CustomOp*>(state)->Compute(api, context);
      return Status::OK();
    };
    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}
}  // namespace onnxruntime

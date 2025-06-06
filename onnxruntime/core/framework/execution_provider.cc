// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/execution_provider.h"
#include "core/framework/execution_providers.h"

#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/murmurhash3.h"
#include "core/framework/op_kernel.h"

#include <stdint.h>

namespace onnxruntime {

std::vector<std::unique_ptr<ComputeCapability>>
IExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                  const IKernelLookup& kernel_lookup,
                                  const GraphOptimizerRegistry&,
                                  IResourceAccountant*) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (const auto& node : graph.Nodes()) {
    if (const KernelCreateInfo* kernel_create_info = kernel_lookup.LookUpKernel(node);
        kernel_create_info != nullptr) {
      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
common::Status IExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& /*fused_nodes_and_graphs*/,
                                           std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
                        "IExecutionProvider::Compile with FusedNodeAndGraph is not implemented by " + type_);
}

#endif

ExecutionProviders::ExecutionProviders() {
#ifdef _WIN32
  // Register callback for ETW capture state (rundown)
  etw_callback_key_ = "ExecutionProviders_rundown_";
  etw_callback_key_.append(std::to_string(reinterpret_cast<uintptr_t>(this)));
  WindowsTelemetry::RegisterInternalCallback(
      etw_callback_key_,
      [this](LPCGUID SourceId,
             ULONG IsEnabled,
             UCHAR Level,
             ULONGLONG MatchAnyKeyword,
             ULONGLONG MatchAllKeyword,
             PEVENT_FILTER_DESCRIPTOR FilterData,
             PVOID CallbackContext) { this->EtwProvidersCallback(SourceId, IsEnabled, Level,
                                                                 MatchAnyKeyword, MatchAllKeyword,
                                                                 FilterData, CallbackContext); });
#endif
}

ExecutionProviders::~ExecutionProviders() {
#ifdef _WIN32
  WindowsTelemetry::UnregisterInternalCallback(etw_callback_key_);
#endif
}

common::Status ExecutionProviders::Add(const std::string& provider_id,
                                       const std::shared_ptr<IExecutionProvider>& p_exec_provider) {
  // make sure there are no issues before we change any internal data structures
  if (provider_idx_map_.find(provider_id) != provider_idx_map_.end()) {
    auto status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Provider ", provider_id, " has already been registered.");
    LOGS_DEFAULT(ERROR) << status.ErrorMessage();
    return status;
  }

  // index that provider will have after insertion
  auto new_provider_idx = exec_providers_.size();

  ORT_IGNORE_RETURN_VALUE(provider_idx_map_.insert({provider_id, new_provider_idx}));

  // update execution provider options
  auto providerOptions = p_exec_provider->GetProviderOptions();
  exec_provider_options_[provider_id] = providerOptions;

#ifdef _WIN32
  LogProviderOptions(provider_id, providerOptions, false);
#endif

  exec_provider_ids_.push_back(provider_id);
  exec_providers_.push_back(p_exec_provider);
  return Status::OK();
}

#ifdef _WIN32
void ExecutionProviders::EtwProvidersCallback(LPCGUID /* SourceId */,
                                              ULONG IsEnabled,
                                              UCHAR /* Level */,
                                              ULONGLONG MatchAnyKeyword,
                                              ULONGLONG /* MatchAllKeyword */,
                                              PEVENT_FILTER_DESCRIPTOR /* FilterData */,
                                              PVOID /* CallbackContext */) {
  // Check if this callback is for capturing state
  if ((IsEnabled == EVENT_CONTROL_CODE_CAPTURE_STATE) &&
      ((MatchAnyKeyword & static_cast<ULONGLONG>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)) != 0)) {
    for (size_t i = 0; i < exec_providers_.size(); ++i) {
      const auto& provider_id = exec_provider_ids_[i];

      auto it = exec_provider_options_.find(provider_id);
      if (it != exec_provider_options_.end()) {
        const auto& options = it->second;

        LogProviderOptions(provider_id, options, true);
      }
    }
  }
}

void ExecutionProviders::LogProviderOptions(const std::string& provider_id,
                                            const ProviderOptions& providerOptions,
                                            bool captureState) {
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  for (const auto& config_pair : providerOptions) {
    TraceLoggingWrite(
        telemetry_provider_handle,
        "ProviderOptions",
        TraceLoggingKeyword(static_cast<uint64_t>(onnxruntime::logging::ORTTraceLoggingKeyword::Session)),
        TraceLoggingLevel(WINEVENT_LEVEL_INFO),
        TraceLoggingString(provider_id.c_str(), "ProviderId"),
        TraceLoggingString(config_pair.first.c_str(), "Key"),
        TraceLoggingString(config_pair.second.c_str(), "Value"),
        TraceLoggingBool(captureState, "isCaptureState"));
  }
#else
  ORT_UNUSED_PARAMETER(provider_id);
  ORT_UNUSED_PARAMETER(providerOptions);
  ORT_UNUSED_PARAMETER(captureState);
#endif
}

#endif

}  // namespace onnxruntime

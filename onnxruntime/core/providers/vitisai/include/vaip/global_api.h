
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/provider_options.h"
#include "core/framework/execution_provider.h"
#include "vaip/my_ort.h"
#include "vaip/dll_safe.h"
#include "vaip/custom_op.h"
#include <optional>
#include <memory>
void initialize_vitisai_ep();
void deinitialize_vitisai_ep();
vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>> compile_onnx_model(const onnxruntime::GraphViewer& graph_viewer, const onnxruntime::logging::Logger& logger, const onnxruntime::ProviderOptions& options);
std::shared_ptr<onnxruntime::KernelRegistry> get_kernel_registry_vitisaiep();
const std::vector<OrtCustomOpDomain*>& get_domains_vitisaiep();
std::optional<std::vector<onnxruntime::Node*>> create_ep_context_nodes(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps);

int vitisai_ep_on_run_start(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps, const void* state,
    vaip_core::DllSafe<std::string> (*get_config_entry)(const void* state, const char* entry_name));
int vitisai_ep_set_ep_dynamic_options(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    const char* const* keys,
    const char* const* values, size_t kv_len);

// Notify EP that profiling has started with the base timestamp (in microseconds since epoch)
// The EP can use this to:
// 1. Calculate relative timestamps (event_ts - base_ts) for the profiling timeline
// 2. Store the absolute base timestamp if needed for other purposes
void profiler_start(uint64_t profiling_start_time_us);

// Notify EP that profiling has stopped
void profiler_stop();

/**
 * EventInfo: Original 5-element tuple (v1 API)
 * Kept for backward compatibility with older vaip versions.
 */
using EventInfo = std::tuple<
    std::string,  // name
    int,          // pid
    int,          // tid
    long long,    // timestamp
    long long     // duration
    >;
void profiler_collect(
    std::vector<EventInfo>& api_events,
    std::vector<EventInfo>& kernel_events);

/**
 * EventInfoV2: Extended 6-element tuple with args map (v2 API)
 * 6th element: args map for extended metadata (subgraph_name, flow_type, kernel_idx)
 */
using EventInfoV2 = std::tuple<
    std::string,                                  // name
    int,                                          // pid
    int,                                          // tid
    long long,                                    // timestamp
    long long,                                    // duration
    std::unordered_map<std::string, std::string>  // args
    >;

// v2 API
// Returns true if the v2 collector symbol was present in the loaded VAIP
// library (i.e. the EP is v2-capable). Returns false if the symbol is not
// available, in which case callers should fall back to profiler_collect (v1).
bool profiler_collect_v2(
    std::vector<EventInfoV2>& api_events,
    std::vector<EventInfoV2>& kernel_events);

std::unique_ptr<onnxruntime::IExecutionProvider>
CreateExecutionProviderFromAnotherEp(const std::string& lib, const OrtSessionOptions& session_options,
                                     std::unordered_map<std::string, std::string>& provider_options);

/**
 * Get compiled model compatibility information from execution providers.
 * Returns a JSON string containing compatibility metadata, or an empty string if unavailable.
 */
std::string get_compiled_model_compatibility_info(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    const onnxruntime::GraphViewer& graph_viewer);

/**
 * Validate compiled model compatibility information against current runtime environment.
 * The model_compatibility is output parameter for the compatibility result.
 */
Status validate_compiled_model_compatibility_info(
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps,
    const std::string& compatibility_info,
    OrtCompiledModelCompatibility& model_compatibility);

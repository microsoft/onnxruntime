
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/provider_options.h"
#include "vaip/my_ort.h"
#include "vaip/dll_safe.h"
#include "vaip/custom_op.h"
#include <optional>
void initialize_vitisai_ep();
vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>> compile_onnx_model(const onnxruntime::GraphViewer& graph_viewer, const onnxruntime::logging::Logger& logger, const onnxruntime::ProviderOptions& options);
std::shared_ptr<onnxruntime::KernelRegistry> get_kernel_registry_vitisaiep();
const std::vector<OrtCustomOpDomain*>& get_domains_vitisaiep();
void get_backend_compilation_cache(const onnxruntime::PathString& model_path_str, const onnxruntime::GraphViewer& graph_viewer, const onnxruntime::ProviderOptions& options, uint8_t compiler_codes, std::string& cache_dir, std::string& cache_key, std::string& cache_data);
void restore_backend_compilation_cache(const std::string& cache_dir, const std::string& cache_key, const std::string& cache_data, const std::string& model_path);
std::optional<std::vector<onnxruntime::Node*>> create_ep_context_nodes(
    onnxruntime::Graph& ep_context_graph,
    const std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>& eps);
bool has_create_ep_context_nodes();

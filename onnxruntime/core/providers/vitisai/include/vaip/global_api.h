
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

void initialize_vitisai_ep();
vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>> compile_onnx_model(const onnxruntime::GraphViewer& graph_viewer, const onnxruntime::logging::Logger& logger, const onnxruntime::ProviderOptions& options);
std::shared_ptr<onnxruntime::KernelRegistry> get_kernel_registry_vitisaiep();
const std::vector<OrtCustomOpDomain*>& get_domains_vitisaiep();

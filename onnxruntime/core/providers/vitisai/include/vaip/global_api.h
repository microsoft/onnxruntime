
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/provider_options.h"
#include <vector>

namespace vaip_core {
class ExecutionProvider;
template <typename T>
class DllSafe;
}  // namespace vaip_core
namespace onnxruntime {
class Graph;
}

std::vector<OrtCustomOpDomain*> initialize_vitisai_ep();
vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>> compile_onnx_model_with_options(const std::string& model_path, const onnxruntime::Graph& graph, const onnxruntime::ProviderOptions& options);

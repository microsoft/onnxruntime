// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <functional>
#include "core/session/onnxruntime_c_api_ep.h"

constexpr const char* kOnnxDomain = "";
constexpr const char* OpenVINO_GPU = "OpenVINO_GPU";
static const std::string OpenVINOEp = "OpenVINOEp";

namespace onnxruntime {
    using NodeIndex = size_t;
    std::string GetEnvironmentVar(const std::string& var_name);
    // TODO(leca): add name (const char*) into OrtValueInfoRef?
    OrtStatus* ForEachNodeDef(const OrtGraphApi* graph_api, const OrtGraphViewer* graph, const OrtNode* node, std::function<void(const char*, const OrtValueInfoRef*, bool/*is_input*/)> func);
}

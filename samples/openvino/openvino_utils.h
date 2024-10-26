// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>

constexpr const char* OpenVINO_GPU = "OpenVINO_GPU";
static const std::string OpenVINOEp = "OpenVINOEp";

namespace onnxruntime {
    std::string GetEnvironmentVar(const std::string& var_name);
}

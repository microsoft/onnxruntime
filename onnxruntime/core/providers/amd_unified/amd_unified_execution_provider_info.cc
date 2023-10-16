// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "./amd_unified_execution_provider_info.h"


namespace onnxruntime {

AMDUnifiedExecutionProviderInfo(const ProviderOptions& provider_option)
    : provider_options_(provider_options) {
}

AMDUnifiedExecutionProviderInfo(const std::string& device_types_str) {
    device_types = ParseDevicesStrRepr(device_types_str);
}

}  // namespace onnxruntime

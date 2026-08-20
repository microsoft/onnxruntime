// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/constants.h"
#include "onnxruntime/core/session/onnxruntime_c_api.h"

namespace onnxruntime::contrib {

// Generated from the com.microsoft schemas at this source commit. A plugin
// embeds this array so ORT can compare the contracts used to build the plugin
// with the live core schema registry when the plugin is loaded.
inline constexpr OrtEpOperatorCompatibilityInfo kMSDomainSchemaAbiManifest[] = {
#include "ms_schema_abi_manifest.inc"
};

}  // namespace onnxruntime::contrib

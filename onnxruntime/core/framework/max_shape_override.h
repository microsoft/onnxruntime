// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

#include "core/common/status.h"
#include "core/framework/resource_accountant.h"  // for MaxShapeOverrideMap

namespace onnxruntime {

/// Parses the session.max_shape_override config string into a name→shape map.
///
/// Format: "name1:[d0,d1,...];name2:[d0,d1,...]"
/// Example: "input_ids:[8,4096];attention_mask:[8,4096]"
///
/// Rules:
/// - Names and dimensions are separated by semicolons.
/// - Each entry is "name:[dim0,dim1,...]".
/// - Dimensions must be positive integers.
/// - Whitespace around names, brackets, and commas is tolerated.
/// - Empty string returns an empty map (success).
///
/// @param config_value  The raw string from the session option.
/// @param out           Populated on success.
/// @return              OK on success, INVALID_ARGUMENT on parse failure.
Status ParseMaxShapeOverride(std::string_view config_value, MaxShapeOverrideMap& out);

}  // namespace onnxruntime

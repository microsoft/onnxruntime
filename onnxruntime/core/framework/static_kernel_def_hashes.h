// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <string>
#include <optional>
#include "core/common/basic_types.h"
namespace onnxruntime {
/**
 * @brief Gets the hash value for provided op type + version combination if it is available, otherwise
 * returns a nullopt. The hash value is available if this node was added by layout transformer. For all other
 * nodes, the hash values should be present either in the serialized session state obtained form ort format model
 * or from compiled kernel hash map which is generated during partitioning.
 * @return std::optional<HashValue>
 */
std::optional<HashValue> GetHashValueFromStaticKernelHashMap(const std::string& op_type, int since_version);
}  // namespace onnxruntime
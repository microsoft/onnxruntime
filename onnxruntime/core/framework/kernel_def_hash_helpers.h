// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/basic_types.h"

#include <optional>
#include <string>

namespace onnxruntime {
class Node;
namespace utils {
/**
 * @brief Gets the hash value for provided op type + version combination if it is available, otherwise
 * returns a nullopt. The hash value is available if this node was added by layout transformer. For all other
 * nodes, the hash values should be present either in the serialized session state obtained form ort format model
 * or from compiled kernel hash map which is generated during partitioning.
 * @return std::optional<HashValue>
 */
std::optional<HashValue> GetHashValueFromStaticKernelHashMap(const std::string& op_type, int since_version);

/**
 * Get hash value for com.microsoft ops with CPU EP implementations that the NHWC optimizer may insert.
 * These are required when that optimizer is run using a minimal build and ORT format model.
 * @param Node Node to find hash for.
 */
std::optional<HashValue> GetInternalNhwcOpHash(const Node& node);

/**
 * Get replacement hash for backwards compatibility if we had to modify an existing kernel registration.
 * @param hash Hash to update if needed.
 */
void UpdateHashForBackwardsCompatibility(HashValue& hash);
}  // namespace utils
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <optional>

namespace onnxruntime {
namespace test {

// Returns the total physical memory (RAM) in bytes, or std::nullopt if detection fails.
std::optional<uint64_t> GetTotalPhysicalMemoryBytes();

}  // namespace test
}  // namespace onnxruntime

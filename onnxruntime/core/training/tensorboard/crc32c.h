// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace onnxruntime {
namespace training {
namespace tensorboard {

uint32_t Crc32cUpdate(uint32_t init_crc, const char* data, size_t size);

// Returns the CRC-32C (Castagnoli) checksum for data[0, size-1] (https://en.wikipedia.org/wiki/Cyclic_redundancy_check)
inline uint32_t Crc32c(const char* data, size_t size) { return Crc32cUpdate(0, data, size); }

}  // namespace tensorboard
}  // namespace training
}  // namespace onnxruntime

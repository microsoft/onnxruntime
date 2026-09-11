// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cstdint>

#include "core/common/status.h"

namespace ONNX_NAMESPACE {
class OpSchema;
}

namespace onnxruntime {

using SchemaAbiDigest = std::array<uint8_t, 32>;

// Computes the versioned, canonical SHA-256 digest of one operator schema.
// Documentation and inference-function implementation details are excluded.
Status ComputeSchemaAbiDigest(const ONNX_NAMESPACE::OpSchema& schema,
                              SchemaAbiDigest& digest);

}  // namespace onnxruntime

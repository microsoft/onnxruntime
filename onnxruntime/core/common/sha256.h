// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/posix/telemetry_sha256.h"

namespace onnxruntime {

// Runtime-wide name for the dependency-free SHA-256 implementation. The
// implementation predates this alias and remains in the telemetry source file
// to avoid an unrelated source move in the plugin schema ABI change.
using Sha256 = telemetry_internal::Sha256;

}  // namespace onnxruntime

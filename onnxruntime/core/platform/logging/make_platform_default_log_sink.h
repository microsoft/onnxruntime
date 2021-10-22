// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/common/logging/isink.h"

namespace onnxruntime {
namespace logging {

/**
 * Creates a log sink that is appropriate for the current platform.
 */
std::unique_ptr<ISink> MakePlatformDefaultLogSink();

}  // namespace logging
}  // namespace onnxruntime

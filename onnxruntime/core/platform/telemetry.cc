// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry.h"
#include "core/platform/env.h"

namespace onnxruntime {

Telemetry::Telemetry() = default;

void LogRuntimeError(uint32_t sessionId, const common::Status& status, const char* file,
                     const char* function, uint32_t line)
{
  const Env& env = Env::Default();
  env.GetTelemetryProvider().LogRuntimeError(sessionId, status, file, function, line);
}

}  // namespace onnxruntime


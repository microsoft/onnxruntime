// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/logging/logging.h"

using onnxruntime::logging::Logger;

struct OrtLogger {
  Logger* ToInternal() { return reinterpret_cast<Logger*>(this); }
  const Logger* ToInternal() const { return reinterpret_cast<const Logger*>(this); }
};

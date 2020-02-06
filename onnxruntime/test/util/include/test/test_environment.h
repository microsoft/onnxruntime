// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/logging/logging.h"

#ifdef HAVE_FRAMEWORK_LIB
#include "core/session/environment.h"
#endif

namespace onnxruntime {
namespace test {

/**
Static logging manager with a CLog based sink so logging macros that use the default logger will work
*/
::onnxruntime::logging::LoggingManager& DefaultLoggingManager();

}  // namespace test
}  // namespace onnxruntime

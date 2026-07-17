// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdlib>

namespace onnxruntime::test {

inline void SuppressTelemetryForTests() {
#ifdef _WIN32
  _putenv_s("ORT_RUNNING_UNIT_TESTS", "1");
#else
  setenv("ORT_RUNNING_UNIT_TESTS", "1", 1);
#endif
}

}  // namespace onnxruntime::test

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdlib>
#include <string>

#include "core/common/common.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <limits.h>
#include <unistd.h>
#endif

namespace onnxruntime {
namespace test {

// Kernels cache some environment switches in function-local statics, so a switch can only be
// changed before the first eligible call in a process. This helper re-runs the selected gtest cases
// in a new process of the current test binary. The caller sets the environment variables (for
// example with ScopedEnvironmentVariables) before calling; the child inherits them. Returns the
// child exit status (0 on success).
//
// Only Windows and Linux are supported (std::system is unavailable on iOS and /proc/self/exe is
// Linux-specific), so include this header only from tests compiled under USE_CUDA.
inline int RunGTestFilterInFreshProcess(const std::string& gtest_filter) {
#ifdef _WIN32
  std::string path(MAX_PATH, '\0');
  const DWORD length = GetModuleFileNameA(nullptr, path.data(), static_cast<DWORD>(path.size()));
  ORT_ENFORCE(length != 0 && length < path.size(), "GetModuleFileNameA failed.");
  path.resize(length);
#else
  std::string path(PATH_MAX, '\0');
  const ssize_t length = readlink("/proc/self/exe", path.data(), path.size());
  ORT_ENFORCE(length > 0 && static_cast<size_t>(length) < path.size(), "readlink(/proc/self/exe) failed.");
  path.resize(static_cast<size_t>(length));
#endif
  const std::string command = "\"" + path + "\" --gtest_filter=" + gtest_filter +
                              " --gtest_fail_if_no_test_selected --gtest_color=no";
  return std::system(command.c_str());
}

}  // namespace test
}  // namespace onnxruntime

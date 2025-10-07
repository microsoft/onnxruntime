// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include <vector>

// USE_LIBBACKTRACE will be defined by CMake for Debug builds when libbacktrace is enabled
#if defined(USE_LIBBACKTRACE)
#include <backtrace.h>
#include <sstream>

namespace onnxruntime {

// A struct to pass data to the libbacktrace callbacks.
struct BacktraceCallbackState {
  std::vector<std::string>* stack;
  int error;
};

// Callback for backtrace_full, called for each frame.
// It receives filename and lineno directly.
int full_callback(void* data, uintptr_t /*pc*/, const char* filename, int lineno, const char* function) {
  auto* state = static_cast<BacktraceCallbackState*>(data);
  std::ostringstream ss;

  // Format the output to be "function_name at file:line"
  ss << (function ? function : "??")
     << " at "
     << (filename ? filename : "??") << ":" << lineno;

  state->stack->push_back(ss.str());
  return 0;  // Return 0 to continue tracing.
}

// Callback for handling errors from libbacktrace.
void error_callback(void* data, const char* /*msg*/, int errnum) {
  auto* state = static_cast<BacktraceCallbackState*>(data);
  state->error = errnum;
}

std::vector<std::string> GetStackTrace() {
  std::vector<std::string> stack;
  BacktraceCallbackState callback_state = {&stack, 0};

  // Create the backtrace state. This initializes the library to read debug info.
  struct backtrace_state* state = backtrace_create_state(nullptr, 0, error_callback, &callback_state);
  if (state == nullptr) {
    return stack;  // Failed to initialize, return empty stack.
  }

  // Perform the full backtrace which calls our callback for each frame.
  // The 'skip' parameter is set to 1 to hide GetStackTrace itself.
  backtrace_full(state, 1, full_callback, error_callback, &callback_state);
  return stack;
}

}  // namespace onnxruntime

#elif !defined(__ANDROID__) && !defined(__wasm__) && !defined(_OPSCHEMA_LIB_) && !defined(_AIX)
#include <execinfo.h>

namespace onnxruntime {
std::vector<std::string> GetStackTrace() {
  std::vector<std::string> stack;

  constexpr int kCallstackLimit = 64;  // Maximum depth of callstack

  void* array[kCallstackLimit];
  char** strings = nullptr;

  int size = backtrace(array, kCallstackLimit);
  stack.reserve(size);
  strings = backtrace_symbols(array, size);

  // NOTE: To get meaningful info from the output, addr2line (or atos on osx) would need to be used.
  // See https://gist.github.com/jvranish/4441299 for an example.
  //
  // To manually translate the output, use the value in the '()' after the executable name with addr2line
  // e.g.
  //   Stacktrace:
  //    /home/me/src/github/onnxruntime/build/Linux/Debug/onnxruntime_test_all(+0x3f46cc) [0x559543faf6cc]
  //
  // >addr2line -f -C -e /home/me/src/github/onnxruntime/build/Linux/Debug/onnxruntime_test_all  +0x3f46cc

  // hide GetStackTrace so the output starts with the 'real' location
  constexpr int start_frame = 1;
  for (int i = start_frame; i < size; i++) {
    stack.push_back(strings[i]);
  }

  free(strings);

  return stack;
}
}  // namespace onnxruntime

#else

namespace onnxruntime {
std::vector<std::string> GetStackTrace() {
  return {};
}
}  // namespace onnxruntime

#endif

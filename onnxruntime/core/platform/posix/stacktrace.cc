// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"

#if !defined(__ANDROID__) && !defined(__wasm__) && !defined(_OPSCHEMA_LIB_)
#include <execinfo.h>
#endif
#include <vector>

namespace onnxruntime {

std::vector<std::string> GetStackTrace() {
  std::vector<std::string> stack;

#if !defined(NDEBUG) && !defined(__ANDROID__) && !defined(__wasm__) && !defined(_OPSCHEMA_LIB_)
  constexpr int kCallstackLimit = 64;  // Maximum depth of callstack

  void* array[kCallstackLimit];
  char** strings = nullptr;

  size_t size = backtrace(array, kCallstackLimit);
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
  constexpr size_t start_frame = 1;
  for (size_t i = start_frame; i < size; i++) {
    stack.push_back(strings[i]);
  }

  free(strings);

#endif

  return stack;
}
}  // namespace onnxruntime

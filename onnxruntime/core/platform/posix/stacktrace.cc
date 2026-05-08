// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include <vector>

#if !defined(NDEBUG) && !defined(__ANDROID__) && !defined(__wasm__) && !defined(_OPSCHEMA_LIB_) && !defined(_AIX)
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <sstream>
#include <unordered_map>
#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"

namespace {

inline int GetAddr2LineEnv() {
  const char* val = std::getenv("ORT_ADDR2LINE");
  if (val == nullptr) {
    return 0;
  }
  return atoi(val);
}

// Check once whether ORT_ENABLE_ADDR2LINE is set.
int GetAddr2LineCount() {
  static const int count = GetAddr2LineEnv();
  return count;
}

// Resolve file offsets to file:line using addr2line, grouped by binary.
// Returns a map from original address to "file:line" string.
std::unordered_map<void*, std::string> ResolveWithAddr2Line(
    void** addresses, int depth) {
  std::unordered_map<void*, std::string> result;

  // Group addresses by their containing binary/shared-object.
  // Key: binary path, Value: vector of (address, file_offset)
  struct FrameInfo {
    void* addr;
    uintptr_t offset;
  };
  std::unordered_map<std::string, std::vector<FrameInfo>> groups;

  for (int i = 0; i < depth; ++i) {
    Dl_info info;
    if (dladdr(addresses[i], &info) && info.dli_fname) {
      uintptr_t offset = reinterpret_cast<uintptr_t>(addresses[i]) -
                         reinterpret_cast<uintptr_t>(info.dli_fbase);
      groups[info.dli_fname].push_back({addresses[i], offset});
    }
  }

  // Call addr2line once per binary with all offsets in batch.
  for (const auto& [binary, frames] : groups) {
    // Use addr2line without -f/-C/-p to get just "file:line" per address.
    std::string cmd = "addr2line -e " + binary;
    for (const auto& frame : frames) {
      char buf[32];
      snprintf(buf, sizeof(buf), " 0x%lx", static_cast<unsigned long>(frame.offset));
      cmd += buf;
    }
    cmd += " 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) continue;

    // Read one line per address from addr2line output.
    char line_buf[1024];
    size_t frame_idx = 0;
    while (fgets(line_buf, sizeof(line_buf), pipe) && frame_idx < frames.size()) {
      // Remove trailing newline
      size_t len = strlen(line_buf);
      if (len > 0 && line_buf[len - 1] == '\n') line_buf[len - 1] = '\0';

      std::string resolved(line_buf);
      // addr2line returns "?? ??:0" or similar for unknown frames — skip those
      if (resolved.find("??") == std::string::npos) {
        result[frames[frame_idx].addr] = resolved;
      }
      ++frame_idx;
    }
    pclose(pipe);
  }

  return result;
}

}  // namespace
#endif

namespace onnxruntime {

std::vector<std::string> GetStackTrace() {
  std::vector<std::string> stack;

#if !defined(NDEBUG) && !defined(__ANDROID__) && !defined(__wasm__) && !defined(_OPSCHEMA_LIB_) && !defined(_AIX)
  constexpr int kCallstackLimit = 64;
  void* addresses[kCallstackLimit];

  // skip_count=2 hides GetStackTrace and its caller (ORT_ENFORCE macro internals)
  int depth = absl::GetStackTrace(addresses, kCallstackLimit, /*skip_count=*/2);
  stack.reserve(depth);

  // Resolve file:line via addr2line only when explicitly opted in via
  // ORT_ENABLE_ADDR2LINE=1, since it spawns external processes and can be
  // very slow on large debug binaries.
  std::unordered_map<void*, std::string> resolved;
  int count = GetAddr2LineCount();
  if (count > 0) {
    resolved = ResolveWithAddr2Line(addresses, std::min(depth, count));
  }

  for (int i = 0; i < depth; ++i) {
    std::ostringstream oss;
    char symbol[1024];
    if (absl::Symbolize(addresses[i], symbol, sizeof(symbol))) {
      oss << symbol;
    } else {
      oss << "[unknown]";
    }

    auto it = resolved.find(addresses[i]);
    if (it != resolved.end()) {
      oss << " at " << it->second;
    } else {
      oss << " [" << addresses[i] << "]";
    }
    stack.push_back(oss.str());
  }
#endif

  return stack;
}
}  // namespace onnxruntime

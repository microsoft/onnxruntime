// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/common/common.h"

#if !defined(NDEBUG) && !defined(__ANDROID__) && !defined(__wasm__) && !defined(_OPSCHEMA_LIB_) && !defined(_AIX)
#include <dlfcn.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <algorithm>
#include <unordered_map>
#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"

extern char** environ;

namespace {

inline int GetAddr2LineEnv() {
  const char* val = std::getenv("ORT_ADDR2LINE");
  if (val == nullptr) {
    return 0;
  }
  return atoi(val);
}

// Check once whether ORT_ADDR2LINE is set.
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
  // Use posix_spawnp + pipe instead of popen to avoid shell injection risks
  // from untrusted binary paths (e.g., paths with spaces or special chars).
  for (const auto& [binary, frames] : groups) {
    // Build argv: {"addr2line", "-e", binary, "0xoffset1", "0xoffset2", ..., nullptr}
    std::vector<std::string> offset_strs;
    offset_strs.reserve(frames.size());
    for (const auto& frame : frames) {
      char buf[32];
      snprintf(buf, sizeof(buf), "0x%lx", static_cast<unsigned long>(frame.offset));
      offset_strs.emplace_back(buf);
    }

    // Build raw argv array for execvp. All pointers reference stable strings above.
    std::vector<char*> argv;
    argv.reserve(3 + frames.size() + 1);
    argv.push_back(const_cast<char*>("addr2line"));
    argv.push_back(const_cast<char*>("-e"));
    argv.push_back(const_cast<char*>(binary.c_str()));
    for (auto& s : offset_strs) {
      argv.push_back(const_cast<char*>(s.c_str()));
    }
    argv.push_back(nullptr);

    // Create a pipe for reading addr2line's stdout.
    int pipe_fds[2];
    if (pipe(pipe_fds) != 0) continue;

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipe_fds[0]);
    // Redirect stderr to /dev/null to suppress error messages.
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);

    pid_t pid;
    int spawn_ret = posix_spawnp(&pid, "addr2line", &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipe_fds[1]);

    if (spawn_ret != 0) {
      close(pipe_fds[0]);
      continue;
    }

    // Read one line per address from addr2line output.
    FILE* fp = fdopen(pipe_fds[0], "r");
    if (!fp) {
      close(pipe_fds[0]);
      waitpid(pid, nullptr, 0);
      continue;
    }

    char line_buf[1024];
    size_t frame_idx = 0;
    while (fgets(line_buf, sizeof(line_buf), fp) && frame_idx < frames.size()) {
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
    fclose(fp);
    waitpid(pid, nullptr, 0);
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
  // ORT_ADDR2LINE=1, since it spawns external processes and can be
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

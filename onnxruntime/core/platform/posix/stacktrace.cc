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
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <algorithm>
#include <unordered_map>
#include "absl/debugging/stacktrace.h"
#include "absl/debugging/symbolize.h"
#include "core/platform/scoped_resource.h"

#ifdef __APPLE__
#include <crt_externs.h>
#define environ (*_NSGetEnviron())
#else
extern char** environ;
#endif

namespace {

// Traits for a POSIX file descriptor, used with onnxruntime::ScopedResource to
// close it on scope exit unless released.
struct FdTraits {
  using Handle = int;
  static Handle GetInvalidHandleValue() noexcept { return -1; }
  static void CleanUp(Handle fd) noexcept { close(fd); }
};
using ScopedFd = onnxruntime::ScopedResource<FdTraits>;

// Traits for a FILE* obtained from fdopen, used with onnxruntime::ScopedResource
// to fclose it on scope exit.
struct FileTraits {
  using Handle = FILE*;
  static Handle GetInvalidHandleValue() noexcept { return nullptr; }
  static void CleanUp(Handle fp) noexcept { fclose(fp); }
};
using ScopedFile = onnxruntime::ScopedResource<FileTraits>;

// RAII wrapper for posix_spawn_file_actions_t.
class ScopedSpawnFileActions {
 public:
  ScopedSpawnFileActions() = default;
  ScopedSpawnFileActions(const ScopedSpawnFileActions&) = delete;
  ScopedSpawnFileActions& operator=(const ScopedSpawnFileActions&) = delete;
  ~ScopedSpawnFileActions() {
    if (initialized_) {
      posix_spawn_file_actions_destroy(&actions_);
    }
  }

  int Init() {
    const int rc = posix_spawn_file_actions_init(&actions_);
    initialized_ = (rc == 0);
    return rc;
  }
  posix_spawn_file_actions_t* get() { return &actions_; }

 private:
  posix_spawn_file_actions_t actions_{};
  bool initialized_ = false;
};

// RAII wrapper that reaps a child process on scope exit and exposes its status.
class ScopedChild {
 public:
  explicit ScopedChild(pid_t pid) : pid_(pid) {}
  ScopedChild(const ScopedChild&) = delete;
  ScopedChild& operator=(const ScopedChild&) = delete;
  ~ScopedChild() { Wait(); }

  // Wait for the child and cache its exit status. Safe to call more than once.
  void Wait() {
    if (pid_ == -1) {
      return;
    }
    int status = 0;
    while (waitpid(pid_, &status, 0) == -1 && errno == EINTR) {
    }
    status_ = status;
    pid_ = -1;
  }

  // Returns true only if the child exited normally with code 0.
  bool ExitedCleanly() {
    Wait();
    return status_.has_value() && WIFEXITED(*status_) && WEXITSTATUS(*status_) == 0;
  }

 private:
  pid_t pid_;
  std::optional<int> status_;
};

// ORT_ADDR2LINE controls optional source file:line resolution for stack frames.
// It is read as a non-negative integer N:
//   - unset or 0 : disabled (default). Only symbol names from absl::Symbolize
//                  are shown and no subprocess is spawned.
//   - N > 0      : resolve file:line for the top N frames by invoking the
//                  external `addr2line` tool (part of binutils). This is opt-in
//                  because spawning addr2line and parsing DWARF can be very slow
//                  on large debug binaries, and addr2line may not be installed.
inline int GetAddr2LineEnv() {
  const char* val = std::getenv("ORT_ADDR2LINE");
  if (val == nullptr) {
    return 0;
  }
  char* end = nullptr;
  long parsed = strtol(val, &end, 10);
  // Reject empty input and any value with trailing non-numeric characters
  // (e.g. "10foo"), so a malformed setting disables resolution rather than
  // silently using a partially parsed number.
  if (end == val || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
    return 0;
  }
  return static_cast<int>(parsed);
}

int GetAddr2LineCount() {
  static const int count = GetAddr2LineEnv();
  return count;
}

bool CreateCloseOnExecPipe(int pipe_fds[2]) {
  if (pipe(pipe_fds) != 0) {
    return false;
  }

  for (int i = 0; i < 2; ++i) {
    const int flags = fcntl(pipe_fds[i], F_GETFD);
    if (flags == -1 || fcntl(pipe_fds[i], F_SETFD, flags | FD_CLOEXEC) == -1) {
      close(pipe_fds[0]);
      close(pipe_fds[1]);
      return false;
    }
  }

  return true;
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
      const uintptr_t addr = reinterpret_cast<uintptr_t>(addresses[i]);
      const uintptr_t base = reinterpret_cast<uintptr_t>(info.dli_fbase);
      // Skip frames whose address is at or below the module base. Subtracting
      // below would underflow uintptr_t and feed a garbage offset to addr2line.
      if (addr <= base) {
        continue;
      }
      // Each captured address is a return address pointing just past the call
      // instruction. Subtract 1 so addr2line resolves the call site itself
      // rather than the following statement (which may be a different source
      // line or even the next function). The map is still keyed by the original
      // address so it matches the absl::Symbolize lookup in GetStackTrace.
      uintptr_t offset = addr - 1 - base;
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
      snprintf(buf, sizeof(buf), "0x%" PRIxPTR, frame.offset);
      offset_strs.emplace_back(buf);
    }

    // Build raw argv array for addr2line. All pointers reference stable strings above.
    std::vector<char*> argv;
    argv.reserve(3 + frames.size() + 1);
    argv.push_back(const_cast<char*>("addr2line"));
    argv.push_back(const_cast<char*>("-e"));
    argv.push_back(const_cast<char*>(binary.c_str()));
    for (auto& s : offset_strs) {
      argv.push_back(const_cast<char*>(s.c_str()));
    }
    argv.push_back(nullptr);

    // Create a pipe for reading addr2line's stdout. RAII wrappers below own the
    // descriptors, spawn file actions, FILE*, and child process so each early
    // 'continue' releases everything without manual cleanup calls.
    int pipe_fds[2];
    if (!CreateCloseOnExecPipe(pipe_fds)) continue;
    ScopedFd read_fd(pipe_fds[0]);
    ScopedFd write_fd(pipe_fds[1]);

    ScopedSpawnFileActions actions;
    if (actions.Init() != 0) continue;

    if (posix_spawn_file_actions_adddup2(actions.get(), write_fd.Get(), STDOUT_FILENO) != 0 ||
        posix_spawn_file_actions_addclose(actions.get(), read_fd.Get()) != 0 ||
        // Redirect stderr to /dev/null to suppress error messages.
        posix_spawn_file_actions_addopen(actions.get(), STDERR_FILENO, "/dev/null", O_WRONLY, 0) != 0) {
      continue;
    }

    pid_t pid = -1;
    const int spawn_ret = posix_spawnp(&pid, "addr2line", actions.get(), nullptr, argv.data(), environ);

    // Close the write end in the parent so we observe EOF once addr2line exits.
    write_fd.Reset();

    // If addr2line cannot be executed (e.g. not installed), glibc posix_spawnp
    // reports the failure here with a non-zero return (ENOENT) and no child is
    // created. Some implementations instead spawn a child that exits 127; that
    // case is caught by the ExitedCleanly() check below, which discards output.
    if (spawn_ret != 0) continue;
    ScopedChild child(pid);

    FILE* raw_fp = fdopen(read_fd.Get(), "r");
    if (raw_fp == nullptr) continue;
    read_fd.Release();  // fp now owns the descriptor.
    ScopedFile fp(raw_fp);

    // Read into a local map and only commit it if addr2line exited cleanly, so a
    // failed run (e.g., addr2line missing, or an unreadable binary) contributes
    // no partial or garbage results.
    std::unordered_map<void*, std::string> binary_result;

    // We intentionally do not use -f/-i so output remains 1 line per address.
    // Adding those flags would break the 1:1 parsing below.
    char line_buf[1024];
    size_t frame_idx = 0;
    while (fgets(line_buf, sizeof(line_buf), fp.Get()) && frame_idx < frames.size()) {
      // Remove trailing newline
      size_t len = strlen(line_buf);
      if (len > 0 && line_buf[len - 1] == '\n') line_buf[len - 1] = '\0';

      std::string resolved(line_buf);
      // addr2line returns "??:0" or "??:?" for unknown frames, so skip those.
      if (resolved != "??:0" && resolved != "??:?" && resolved.substr(0, 2) != "??") {
        binary_result[frames[frame_idx].addr] = std::move(resolved);
      }
      ++frame_idx;
    }

    // Reap the child and only trust the output if it exited with status 0.
    if (child.ExitedCleanly()) {
      for (auto& kv : binary_result) {
        result.insert(std::move(kv));
      }
    }
  }

  return result;
}

}  // namespace
#endif  // !defined(NDEBUG) && !defined(__ANDROID__) && !defined(__wasm__) && !defined(_OPSCHEMA_LIB_) && !defined(_AIX)

namespace onnxruntime {

std::vector<std::string> GetStackTrace() {
  std::vector<std::string> stack;

#if !defined(NDEBUG) && !defined(__ANDROID__) && !defined(__wasm__) && !defined(_OPSCHEMA_LIB_) && !defined(_AIX)
  constexpr int kCallstackLimit = 64;
  void* addresses[kCallstackLimit];

  // skip_count=2 hides GetStackTrace and its caller (ORT_ENFORCE macro internals)
  int depth = absl::GetStackTrace(addresses, kCallstackLimit, /*skip_count=*/2);
  stack.reserve(depth);

  // Resolve file:line via addr2line only when explicitly opted in via ORT_ADDR2LINE=*,
  // since it spawns external processes and can be very slow on large debug binaries.
  std::unordered_map<void*, std::string> resolved;
  int count = GetAddr2LineCount();
  if (count > 0) {
    // Only resolve the top N frames to limit addr2line overhead on large binaries.
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
#endif  // !defined(NDEBUG) && !defined(__ANDROID__) && !defined(__wasm__) && !defined(_OPSCHEMA_LIB_) && !defined(_AIX)

  return stack;
}
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/device_id.h"

#include "core/common/common.h"
#include "core/platform/telemetry_guid.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <pwd.h>
#if defined(__has_include)
#if __has_include(<sys/file.h>)
#include <sys/file.h>
#define ORT_DEVICE_ID_USE_FLOCK 1
#endif
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace onnxruntime {

namespace {

// ORT and OGA share this file and use the same flock-first protocol so corruption repair is
// serialized even when both libraries are loaded in one process.
constexpr char kDeviceIdLockFileName[] = "deviceid.lock";

enum class DeviceIdReadResult {
  Missing,
  Read,
  Invalid,
  Failed,
};

struct DeviceIdFileRead {
  DeviceIdReadResult result;
  std::string content;
};

class ScopedFileDescriptor {
 public:
  explicit ScopedFileDescriptor(int fd = -1) : fd_(fd) {}
  ~ScopedFileDescriptor() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  ScopedFileDescriptor(const ScopedFileDescriptor&) = delete;
  ScopedFileDescriptor& operator=(const ScopedFileDescriptor&) = delete;

  int Get() const { return fd_; }
  explicit operator bool() const { return fd_ >= 0; }

 private:
  int fd_;
};

ScopedFileDescriptor OpenStorageDirectoryNoFollow(const std::string& path) {
  int flags = O_RDONLY;
#ifdef O_DIRECTORY
  flags |= O_DIRECTORY;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif

  const int fd = ::open(path.c_str(), flags);
  if (fd < 0) {
    return ScopedFileDescriptor{};
  }

  struct stat directory_info{};
  if (::fstat(fd, &directory_info) != 0 ||
      !S_ISDIR(directory_info.st_mode) ||
      directory_info.st_uid != ::geteuid() ||
      ::fchmod(fd, S_IRWXU) != 0) {
    ::close(fd);
    return ScopedFileDescriptor{};
  }

  return ScopedFileDescriptor(fd);
}

bool AcquireExclusiveFileLock(int fd) {
  constexpr auto kRetryDelay = std::chrono::milliseconds(10);
  constexpr auto kTimeout = std::chrono::seconds(1);
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;

  while (true) {
#if defined(ORT_DEVICE_ID_USE_FLOCK)
    if (::flock(fd, LOCK_EX | LOCK_NB) == 0) {
      return true;
    }
#else
    struct flock lock{};
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    if (::fcntl(fd, F_SETLK, &lock) == 0) {
      return true;
    }
#endif

    if (errno == EINTR) {
      continue;
    }
    if (errno != EACCES && errno != EAGAIN && errno != EWOULDBLOCK) {
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(kRetryDelay);
  }
}

class ScopedDeviceIdFileLock {
 public:
  explicit ScopedDeviceIdFileLock(int directory_fd) {
    int flags = O_RDWR | O_CREAT;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif

    fd_ = ::openat(directory_fd, kDeviceIdLockFileName, flags, S_IRUSR | S_IWUSR);
    if (fd_ < 0) {
      return;
    }

    struct stat file_info{};
    if (::fstat(fd_, &file_info) != 0 ||
        !S_ISREG(file_info.st_mode) ||
        file_info.st_uid != ::geteuid() ||
        ::fchmod(fd_, S_IRUSR | S_IWUSR) != 0 ||
        !AcquireExclusiveFileLock(fd_)) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  ~ScopedDeviceIdFileLock() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  ScopedDeviceIdFileLock(const ScopedDeviceIdFileLock&) = delete;
  ScopedDeviceIdFileLock& operator=(const ScopedDeviceIdFileLock&) = delete;

  explicit operator bool() const { return fd_ >= 0; }

 private:
  int fd_{-1};
};

void TrimAsciiWhitespace(std::string& value) {
  value.erase(std::find_if_not(value.rbegin(), value.rend(),
                               [](unsigned char c) { return std::isspace(c); })
                  .base(),
              value.end());
  value.erase(value.begin(), std::find_if_not(value.begin(), value.end(),
                                              [](unsigned char c) { return std::isspace(c); }));
}

DeviceIdFileRead ReadDeviceIdFileNoFollow(int directory_fd, const char* file_name, size_t max_size) {
  int flags = O_RDONLY;
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
#ifdef O_NONBLOCK
  flags |= O_NONBLOCK;
#endif
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif

  const int fd = ::openat(directory_fd, file_name, flags);
  if (fd < 0) {
    return {errno == ENOENT ? DeviceIdReadResult::Missing : DeviceIdReadResult::Failed, {}};
  }

  struct stat file_info{};
  if (::fstat(fd, &file_info) != 0 || !S_ISREG(file_info.st_mode)) {
    ::close(fd);
    return {DeviceIdReadResult::Failed, {}};
  }

  std::vector<char> buffer(max_size + 1);
  size_t total = 0;
  bool read_ok = true;
  while (total < buffer.size()) {
    const ssize_t count = ::read(fd, buffer.data() + total, buffer.size() - total);
    if (count > 0) {
      total += static_cast<size_t>(count);
    } else if (count == 0) {
      break;
    } else if (errno != EINTR) {
      read_ok = false;
      break;
    }
  }
  if (::close(fd) != 0) read_ok = false;
  if (!read_ok) return {DeviceIdReadResult::Failed, {}};
  if (total > max_size) return {DeviceIdReadResult::Invalid, {}};

  std::string content(buffer.data(), total);
  TrimAsciiWhitespace(content);
  return {DeviceIdReadResult::Read, std::move(content)};
}

}  // namespace

DeviceId& DeviceId::Instance() {
  static DeviceId instance;
  return instance;
}

std::string DeviceId::GetValue() {
  std::lock_guard<std::mutex> lock(mutex_);
  InitializeInternal();
  return device_id_;
}

DeviceIdStatus DeviceId::GetStatus() {
  std::lock_guard<std::mutex> lock(mutex_);
  InitializeInternal();
  return status_;
}

std::string DeviceId::GetStatusString() {
  switch (GetStatus()) {
    case DeviceIdStatus::New:
      return "New";
    case DeviceIdStatus::Existing:
      return "Existing";
    case DeviceIdStatus::Corrupted:
      return "Corrupted";
    case DeviceIdStatus::Failed:
      return "Failed";
    default:
      return "Unknown";
  }
}

bool DeviceId::IsValidGUID(const std::string& str) {
  if (str.length() != 36) return false;

  for (size_t i = 0; i < str.length(); ++i) {
    char c = str[i];
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (c != '-') return false;
    } else {
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
        return false;
      }
    }
  }
  return true;
}

std::string DeviceId::GetStorageDirectory() {
#if !defined(__APPLE__)
  // XDG requires absolute paths. Ignore relative values so telemetry state is never written below
  // the process working directory.
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && xdg[0] == '/') {
    return std::string(xdg) + "/" + kDeviceIdDir;
  }
#endif

  // Prefer an absolute $HOME; fall back to the password database for contexts where HOME is unset
  // or invalid, e.g. system services/daemons under systemd/launchd.
  std::string home;
  if (const char* h = std::getenv("HOME"); h != nullptr && h[0] == '/') {
    home = h;
  } else {
    // getpwuid() returns a pointer to shared static storage and is not thread-safe; use the
    // reentrant getpwuid_r() with a caller-provided buffer so concurrent callers don't race.
    struct passwd pwd;
    struct passwd* result = nullptr;
    const long sc = ::sysconf(_SC_GETPW_R_SIZE_MAX);
    constexpr size_t kDefaultPwBufferSize = 16384;
    constexpr size_t kMaxPwBufferSize = 1 << 20;
    const size_t pw_buffer_size =
        sc > 0 ? std::min(static_cast<size_t>(sc), kMaxPwBufferSize) : kDefaultPwBufferSize;
    std::vector<char> buf(pw_buffer_size);
    if (::getpwuid_r(::getuid(), &pwd, buf.data(), buf.size(), &result) == 0 &&
        result != nullptr && result->pw_dir != nullptr && result->pw_dir[0] == '/') {
      home = result->pw_dir;
    }
  }
  if (home.empty()) return "";

#if defined(__APPLE__)
  return home + "/Library/Application Support/" + kDeviceIdDir;
#else
  return home + "/.cache/" + kDeviceIdDir;
#endif
}

std::string DeviceId::EnsureStorageDirectory() {
  std::string dir = GetStorageDirectory();
  if (!dir.empty() && !CreateDirectoryTree(dir)) {
    return "";
  }
  return dir;
}

bool DeviceId::CreateDirectoryTree(const std::string& path, bool leaf) {
  if (path.empty()) return false;

  struct stat path_info{};
  if (::lstat(path.c_str(), &path_info) == 0) {
    if (leaf && S_ISLNK(path_info.st_mode)) {
      return false;
    }
    if (::stat(path.c_str(), &path_info) != 0 || !S_ISDIR(path_info.st_mode)) {
      return false;
    }
    if (leaf) {
      if (::chmod(path.c_str(), S_IRWXU) != 0) {
        return false;
      }
    }
    return true;
  }
  if (errno != ENOENT) {
    return false;
  }

  size_t pos = path.find_last_of('/');
  if (pos != std::string::npos && pos > 0) {
    if (!CreateDirectoryTree(path.substr(0, pos), false)) {
      return false;
    }
  }

  if (::mkdir(path.c_str(), S_IRWXU) != 0 && errno != EEXIST) {
    return false;
  }
  if (::lstat(path.c_str(), &path_info) != 0 || (leaf && S_ISLNK(path_info.st_mode))) {
    return false;
  }
  if (::stat(path.c_str(), &path_info) != 0 || !S_ISDIR(path_info.st_mode)) {
    return false;
  }
  return !leaf || ::chmod(path.c_str(), S_IRWXU) == 0;
}

void DeviceId::InitializeInternal() {
  if (initialized_) return;
  initialized_ = true;

  ORT_TRY {
    // Keep an ephemeral fallback so persistence failures never expose the SDK's hardware-derived
    // desktop identifier.
    device_id_ = GenerateGuidV4();

    std::string dir_path = GetStorageDirectory();
    if (dir_path.empty()) {
      status_ = DeviceIdStatus::Failed;
      return;
    }

    if (!CreateDirectoryTree(dir_path)) {
      status_ = DeviceIdStatus::Failed;
      return;
    }
    ScopedFileDescriptor directory = OpenStorageDirectoryNoFollow(dir_path);
    if (!directory) {
      status_ = DeviceIdStatus::Failed;
      return;
    }

    // Try to read existing device ID
    const DeviceIdFileRead existing = ReadDeviceIdFileNoFollow(directory.Get(), kFileName, kMaxFileSize);
    if (existing.result == DeviceIdReadResult::Read && IsValidGUID(existing.content)) {
      device_id_ = existing.content;
      status_ = DeviceIdStatus::Existing;
      return;
    }
    if (existing.result == DeviceIdReadResult::Failed) {
      status_ = DeviceIdStatus::Failed;
      return;
    }
    if (existing.result != DeviceIdReadResult::Missing) {
      status_ = DeviceIdStatus::Corrupted;
    }

    const bool regenerated_from_corruption = (status_ == DeviceIdStatus::Corrupted);
    std::optional<ScopedDeviceIdFileLock> repair_lock;
    if (regenerated_from_corruption) {
      repair_lock.emplace(directory.Get());
      if (!*repair_lock) {
        status_ = DeviceIdStatus::Failed;
        return;
      }

      // Another ORT or OGA process may have repaired the shared file while this process waited.
      const DeviceIdFileRead repaired = ReadDeviceIdFileNoFollow(directory.Get(), kFileName, kMaxFileSize);
      if (repaired.result == DeviceIdReadResult::Read && IsValidGUID(repaired.content)) {
        device_id_ = repaired.content;
        status_ = DeviceIdStatus::Existing;
        return;
      }
      if (repaired.result == DeviceIdReadResult::Failed) {
        status_ = DeviceIdStatus::Failed;
        return;
      }
    }

    const std::string temp_name = std::string(kFileName) + ".tmp." + GenerateGuidV4();
    int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = ::openat(directory.Get(), temp_name.c_str(), flags, S_IRUSR | S_IWUSR);
    if (fd >= 0) {
      bool wrote = ::fchmod(fd, S_IRUSR | S_IWUSR) == 0;
      const char* data = device_id_.data();
      size_t remaining = device_id_.size();
      while (wrote && remaining > 0) {
        const ssize_t written = ::write(fd, data, remaining);
        if (written < 0 && errno == EINTR) {
          continue;
        }
        if (written <= 0) {
          wrote = false;
          break;
        }
        data += written;
        remaining -= static_cast<size_t>(written);
      }
      const int close_result = ::close(fd);
      wrote = wrote && remaining == 0 && close_result == 0;
      if (!wrote) {
        ::unlinkat(directory.Get(), temp_name.c_str(), 0);
        status_ = DeviceIdStatus::Failed;
      } else if (regenerated_from_corruption) {
        if (::renameat(directory.Get(), temp_name.c_str(), directory.Get(), kFileName) == 0) {
          status_ = DeviceIdStatus::Corrupted;
        } else {
          ::unlinkat(directory.Get(), temp_name.c_str(), 0);
          status_ = DeviceIdStatus::Failed;
        }
      } else {
        const int link_result = ::linkat(directory.Get(), temp_name.c_str(), directory.Get(), kFileName, 0);
        const int link_error = errno;
        ::unlinkat(directory.Get(), temp_name.c_str(), 0);
        if (link_result == 0) {
          status_ = DeviceIdStatus::New;
        } else if (link_error == EEXIST) {
          // Another process won the first-run race. Its complete file was published atomically,
          // so use that value instead of allowing the persisted id to flap.
          const DeviceIdFileRead winner = ReadDeviceIdFileNoFollow(directory.Get(), kFileName, kMaxFileSize);
          if (winner.result == DeviceIdReadResult::Read && IsValidGUID(winner.content)) {
            device_id_ = winner.content;
            status_ = DeviceIdStatus::Existing;
          } else {
            status_ = DeviceIdStatus::Failed;
          }
        } else {
          status_ = DeviceIdStatus::Failed;
        }
      }
    } else {
      status_ = DeviceIdStatus::Failed;
    }
  }
  ORT_CATCH(...) {
    status_ = DeviceIdStatus::Failed;
    // Keep device_id_ if generated — it's still valid for this session (in-memory only).
  }
}

}  // namespace onnxruntime

#if defined(ORT_DEVICE_ID_USE_FLOCK)
#undef ORT_DEVICE_ID_USE_FLOCK
#endif

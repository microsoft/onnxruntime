// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/device_id.h"

#include "core/common/common.h"
#include "core/platform/telemetry_guid.h"

#include <algorithm>
#include <fstream>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#include <fcntl.h>

namespace onnxruntime {

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
  // Prefer $HOME; fall back to the password database (getpwuid) for contexts where HOME is unset,
  // e.g. system services/daemons under systemd/launchd.
  std::string home;
  if (const char* h = std::getenv("HOME"); h != nullptr && h[0] != '\0') {
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
        result != nullptr && result->pw_dir != nullptr && result->pw_dir[0] != '\0') {
      home = result->pw_dir;
    }
  }
  if (home.empty()) return "";

#if defined(__APPLE__)
  return home + "/Library/Application Support/" + kDeviceIdDir;
#else
  // Follow the XDG Base Directory spec: prefer $XDG_CACHE_HOME, otherwise ~/.cache.
  std::string cache_base;
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && xdg[0] != '\0') {
    cache_base = xdg;
  } else {
    cache_base = home + "/.cache";
  }
  return cache_base + "/" + kDeviceIdDir;
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
      ::chmod(path.c_str(), S_IRWXU);
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
  return ::stat(path.c_str(), &path_info) == 0 && S_ISDIR(path_info.st_mode);
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

    std::string file_path = dir_path + "/" + kFileName;

    // Try to read existing device ID
    {
      struct stat file_info{};
      if (::lstat(file_path.c_str(), &file_info) == 0 && S_ISLNK(file_info.st_mode)) {
        status_ = DeviceIdStatus::Failed;
        return;
      }
      std::ifstream infile(file_path);
      if (infile.good()) {
        infile.seekg(0, std::ios::end);
        auto size = infile.tellg();
        infile.seekg(0, std::ios::beg);

        if (size > static_cast<std::streamoff>(kMaxFileSize)) {
          status_ = DeviceIdStatus::Corrupted;
        } else {
          std::string content;
          std::getline(infile, content);

          // Trim whitespace
          while (!content.empty() &&
                 (content.back() == '\n' || content.back() == '\r' || content.back() == ' ')) {
            content.pop_back();
          }

          if (IsValidGUID(content)) {
            device_id_ = content;
            status_ = DeviceIdStatus::Existing;
            return;
          }
          status_ = DeviceIdStatus::Corrupted;
        }
      }
    }

    // Create directory tree
    if (!CreateDirectoryTree(dir_path)) {
      status_ = DeviceIdStatus::Failed;
      return;
    }
    const bool regenerated_from_corruption = (status_ == DeviceIdStatus::Corrupted);
    const std::string temp_path = file_path + ".tmp." + GenerateGuidV4();
    int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = ::open(temp_path.c_str(), flags, S_IRUSR | S_IWUSR);
    if (fd >= 0) {
      ::fchmod(fd, S_IRUSR | S_IWUSR);
      const char* data = device_id_.data();
      size_t remaining = device_id_.size();
      while (remaining > 0) {
        const ssize_t written = ::write(fd, data, remaining);
        if (written <= 0) {
          break;
        }
        data += written;
        remaining -= static_cast<size_t>(written);
      }
      const int close_result = ::close(fd);
      const bool wrote = remaining == 0 && close_result == 0;
      if (!wrote) {
        ::unlink(temp_path.c_str());
        status_ = DeviceIdStatus::Failed;
      } else if (regenerated_from_corruption) {
        if (::rename(temp_path.c_str(), file_path.c_str()) == 0) {
          status_ = DeviceIdStatus::Corrupted;
        } else {
          ::unlink(temp_path.c_str());
          status_ = DeviceIdStatus::Failed;
        }
      } else {
        const int link_result = ::link(temp_path.c_str(), file_path.c_str());
        const int link_error = errno;
        ::unlink(temp_path.c_str());
        if (link_result == 0) {
          status_ = DeviceIdStatus::New;
        } else if (link_error == EEXIST) {
          // Another process won the first-run race. Its complete file was published atomically,
          // so use that value instead of allowing the persisted id to flap.
          struct stat file_info{};
          std::ifstream winner;
          if (::lstat(file_path.c_str(), &file_info) == 0 && !S_ISLNK(file_info.st_mode)) {
            winner.open(file_path);
          }
          std::string content;
          if (winner.good() && std::getline(winner, content)) {
            while (!content.empty() &&
                   (content.back() == '\n' || content.back() == '\r' || content.back() == ' ')) {
              content.pop_back();
            }
          }
          if (IsValidGUID(content)) {
            device_id_ = content;
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

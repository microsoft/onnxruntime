// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/device_id.h"

#include "core/common/common.h"
#include "core/platform/telemetry_guid.h"

#include <fstream>
#include <cstdint>
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
    std::vector<char> buf(sc > 0 ? static_cast<size_t>(sc) : 16384);
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
  if (!dir.empty()) {
    CreateDirectoryTree(dir);
  }
  return dir;
}

void DeviceId::CreateDirectoryTree(const std::string& path) {
  if (path.empty()) return;

  size_t pos = path.find_last_of('/');
  if (pos != std::string::npos && pos > 0) {
    CreateDirectoryTree(path.substr(0, pos));
  }

  // Owner-only (0700): this tree holds the persistent device id and the telemetry offline cache, so
  // it should not be listable/traversable by other users. mkdir only sets the mode for directories
  // it actually creates; pre-existing directories are left untouched.
  mkdir(path.c_str(), 0700);
}

void DeviceId::InitializeInternal() {
  if (initialized_) return;
  initialized_ = true;

  ORT_TRY {
    std::string dir_path = GetStorageDirectory();
    if (dir_path.empty()) {
      status_ = DeviceIdStatus::Failed;
      return;
    }

    std::string file_path = dir_path + "/" + kFileName;

    // Try to read existing device ID
    {
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

    // Generate new device ID
    device_id_ = GenerateGuidV4();

    // Create directory tree
    CreateDirectoryTree(dir_path);

    // Persist with owner-only (0600) permissions from creation. Using open() with mode 0600 (and
    // fchmod to also tighten a pre-existing file) avoids the window where std::ofstream would create
    // the file using the process umask and only chmod it afterwards — during which the device id
    // could briefly be world-readable. fchmod runs before any write, so content is never exposed.
    const bool regenerated_from_corruption = (status_ == DeviceIdStatus::Corrupted);
    const int fd = ::open(file_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd >= 0) {
      ::fchmod(fd, S_IRUSR | S_IWUSR);
      const ssize_t written = ::write(fd, device_id_.data(), device_id_.size());
      ::close(fd);
      if (written == static_cast<ssize_t>(device_id_.size())) {
        // Preserve Corrupted (defined as "invalid and regenerated") instead of overwriting it with
        // New, so callers/telemetry can still observe that the persisted id had to be regenerated.
        status_ = regenerated_from_corruption ? DeviceIdStatus::Corrupted : DeviceIdStatus::New;
      } else {
        status_ = DeviceIdStatus::Failed;
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

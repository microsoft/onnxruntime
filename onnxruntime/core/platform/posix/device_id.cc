// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/device_id.h"

#include <fstream>
#include <sstream>
#include <random>
#include <iomanip>
#include <cstdlib>

#include <sys/stat.h>
#include <sys/types.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

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

std::string DeviceId::GenerateUUID() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);

  uint32_t data1 = dist(gen);
  uint16_t data2 = static_cast<uint16_t>(dist(gen) & 0xFFFF);
  uint16_t data3 = static_cast<uint16_t>((dist(gen) & 0x0FFF) | 0x4000);  // Version 4
  uint16_t data4 = static_cast<uint16_t>((dist(gen) & 0x3FFF) | 0x8000);  // Variant 1
  uint16_t data5a = static_cast<uint16_t>(dist(gen) & 0xFFFF);
  uint32_t data5b = dist(gen);

  std::ostringstream oss;
  oss << std::hex << std::setfill('0')
      << std::setw(8) << data1 << '-'
      << std::setw(4) << data2 << '-'
      << std::setw(4) << data3 << '-'
      << std::setw(4) << data4 << '-'
      << std::setw(4) << data5a
      << std::setw(8) << data5b;
  return oss.str();
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

std::string DeviceId::GetStorageDirectory(bool mobile) {
  const char* h = std::getenv("HOME");
  if (!h || !h[0]) return "";
  std::string home(h);

  if (mobile) {
    return home + "/.onnxruntime";
  }

#if defined(__APPLE__)
  return home + "/Library/Application Support/" + kDeviceIdDir;
#else
  return home + "/" + kDeviceIdDir;
#endif
}

void DeviceId::CreateDirectoryTree(const std::string& path) {
  if (path.empty()) return;

  size_t pos = path.find_last_of('/');
  if (pos != std::string::npos && pos > 0) {
    CreateDirectoryTree(path.substr(0, pos));
  }

  mkdir(path.c_str(), 0755);
}

void DeviceId::InitializeInternal() {
  if (initialized_) return;
  initialized_ = true;

  try {
    // Use compile-time platform detection to select the appropriate storage path.
    // This matches the mobile/desktop selection in posix/env.cc.
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IOS)
    constexpr bool is_mobile = true;
#else
    constexpr bool is_mobile = false;
#endif
    std::string dir_path = GetStorageDirectory(is_mobile);
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
    device_id_ = GenerateUUID();

    // Create directory tree
    CreateDirectoryTree(dir_path);

    // Write to file
    std::ofstream outfile(file_path);
    if (outfile.good()) {
      outfile << device_id_;
      outfile.close();
      status_ = DeviceIdStatus::New;
    } else {
      status_ = DeviceIdStatus::Failed;
    }
  } catch (...) {
    status_ = DeviceIdStatus::Failed;
    // Keep device_id_ if generated — it's still valid for this session (in-memory only).
  }
}

}  // namespace onnxruntime

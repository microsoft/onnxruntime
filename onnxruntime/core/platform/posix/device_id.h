// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <mutex>
#include "core/common/common.h"

namespace onnxruntime {

enum class DeviceIdStatus {
  New,        // Device ID was newly generated
  Existing,   // Device ID was loaded from persistent storage
  Corrupted,  // Stored device ID was invalid and regenerated
  Failed      // Failed to persist device ID (in-memory only)
};

/**
 * Manages a persistent device identifier for telemetry purposes.
 * The device ID is stored in a platform-appropriate location:
 * - macOS: ~/Library/Application Support/Microsoft/DeveloperTools/.onnxruntime/deviceid
 * - Linux: ~/Microsoft/DeveloperTools/.onnxruntime/deviceid
 * - iOS/Android: ~/.onnxruntime/deviceid (shorter path, avoids iCloud backup on iOS)
 *
 * Thread-safe singleton - use DeviceId::Instance() to access.
 */
class DeviceId {
 public:
  static DeviceId& Instance();

  // Get the device ID value (generates/loads on first call)
  std::string GetValue();

  // Get the status of the device ID
  DeviceIdStatus GetStatus();

  // Get human-readable status string
  std::string GetStatusString();

  // Get the directory path for device ID / telemetry cache storage
  // Desktop: ~/Microsoft/DeveloperTools/.onnxruntime (or platform equivalent)
  // Mobile: ~/.onnxruntime
  static std::string GetStorageDirectory(bool mobile = false);

 private:
  DeviceId() = default;
  ~DeviceId() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DeviceId);

  void InitializeInternal();

  // Generate a random UUID v4
  static std::string GenerateUUID();

  // Validate GUID format (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
  static bool IsValidGUID(const std::string& str);

  // Create directory tree recursively using platform APIs
  static void CreateDirectoryTree(const std::string& path);

  static constexpr const char* kDeviceIdDir = "Microsoft/DeveloperTools/.onnxruntime";
  static constexpr const char* kFileName = "deviceid";
  static constexpr size_t kMaxFileSize = 256;

  std::string device_id_;
  DeviceIdStatus status_ = DeviceIdStatus::New;
  bool initialized_ = false;
  std::mutex mutex_;
};
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include "core/common/logging/isink.h"

namespace onnxruntime {
namespace logging {

/// <summary>
/// A logging sink that writes to a file.
/// </summary>
/// <seealso cref="ISink" />
class FileSink : public ISink {
 public:
  FileSink(const std::filesystem::path& file_path, bool append = false, bool filter_user_data = false);
  ~FileSink() override;

  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override;

 private:
  FILE* file_{};
  bool filter_user_data_;
  bool flush_on_each_write_{true};
  std::mutex mutex_;
};

}  // namespace logging
}  // namespace onnxruntime
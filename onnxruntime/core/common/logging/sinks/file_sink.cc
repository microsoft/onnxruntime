// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/logging/sinks/file_sink.h"

#include <sstream>
#include <stdexcept>

#include "core/platform/env.h"

namespace onnxruntime {
namespace logging {

FileSink::FileSink(const std::filesystem::path& file_path, bool append, bool filter_user_data)
    : filter_user_data_(filter_user_data) {
  const wchar_t* mode = append ? L"a" : L"w";
#ifdef _WIN32
  if (_wfopen_s(&file_, file_path.c_str(), mode) != 0) {
    file_ = nullptr;
  }
#else
  file_ = fopen(file_path.c_str(), append ? "a" : "w");
#endif

  if (file_ == nullptr) {
    throw std::runtime_error("Failed to open log file " + file_path.string());
  }
}

FileSink::~FileSink() {
  if (file_ != nullptr) {
    fclose(file_);
  }
}

void FileSink::SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) {
  if (!filter_user_data_ || message.DataType() != DataType::USER) {
    std::ostringstream msg_stream;
    using timestamp_ns::operator<<;
    timestamp_ns::operator<<(msg_stream, timestamp);

    msg_stream << " [" << message.SeverityPrefix() << ":" << message.Category() << ":" << logger_id << ", "
               << message.Location().ToString() << "] " << message.Message() << "\n";

    std::string message_to_log = msg_stream.str();

    std::lock_guard<std::mutex> lock(mutex_);
    if (file_) {
      fprintf(file_, "%s", message_to_log.c_str());

      if (flush_on_each_write_) {
        fflush(file_);
      }
    }
  }
}

}  // namespace logging
}  // namespace onnxruntime
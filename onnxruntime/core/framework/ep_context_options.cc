// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <limits>
#include <string>
#include <utility>
#include "core/common/common.h"
#include "core/framework/ep_context_options.h"
#include "core/framework/error_code_helper.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace epctx {
// class ModelGenOptions

ModelGenOptions::ModelGenOptions(const ConfigOptions& config_options) {
  enable = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEnable, "0") == "1";

  std::string output_model_path = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");
  if (!output_model_path.empty()) {
    output_model_location = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextFilePath, "");
  } else {
    output_model_location = std::monostate{};
  }

  std::string external_initializers_file_path = config_options.GetConfigOrDefault(
      kOrtSessionOptionsEpContextModelExternalInitializersFileName, "");
  if (!external_initializers_file_path.empty()) {
    ExternalInitializerFileInfo ext_info = {};
    ext_info.file_path = external_initializers_file_path;
    ext_info.size_threshold = 0;
    initializers_location = std::move(ext_info);
  }

  embed_ep_context_in_model = config_options.GetConfigOrDefault(kOrtSessionOptionEpContextEmbedMode, "0") == "1";
}

bool ModelGenOptions::HasOutputModelLocation() const {
  return !std::holds_alternative<std::monostate>(output_model_location);
}

const std::string* ModelGenOptions::TryGetOutputModelPath() const {
  return std::get_if<std::string>(&output_model_location);
}

const BufferHolder* ModelGenOptions::TryGetOutputModelBuffer() const {
  return std::get_if<BufferHolder>(&output_model_location);
}

const OutStreamHolder* ModelGenOptions::TryGetOutputModelOutStream() const {
  return std::get_if<OutStreamHolder>(&output_model_location);
}

bool ModelGenOptions::AreInitializersEmbeddedInOutputModel() const {
  return std::holds_alternative<std::monostate>(initializers_location);
}

const ExternalInitializerFileInfo* ModelGenOptions::TryGetExternalInitializerFileInfo() const {
  return std::get_if<ExternalInitializerFileInfo>(&initializers_location);
}

const InitializerHandler* ModelGenOptions::TryGetInitializerHandler() const {
  return std::get_if<InitializerHandler>(&initializers_location);
}

// class OutStreamBuf

OutStreamBuf::OutStreamBuf(OutStreamHolder out_stream_holder) : out_stream_holder_(out_stream_holder) {
  setp(buffer_.data(), buffer_.data() + buffer_.size());
}

OutStreamBuf::~OutStreamBuf() {
  sync();
}

// Called when the buffer_ is full. Flushes the buffer_ (via sync()) and then writes the overflow character to buffer_.
std::streambuf::int_type OutStreamBuf::overflow(std::streambuf::int_type ch) {
  if (sync() == -1) {
    return traits_type::eof();
  }

  if (ch != traits_type::eof()) {
    *pptr() = static_cast<char>(ch);
    pbump(1);
  }

  return ch;
}

// Flushes the entire buffer_ to the user's write function.
int OutStreamBuf::sync() {
  if (!last_status_.IsOK()) {
    return -1;
  }

  std::ptrdiff_t num_bytes = pptr() - pbase();
  if (num_bytes == 0) {
    return 0;
  }

  // Can only call pbump() with an int, so can only write at most (2^31 - 1) bytes.
  if (num_bytes > std::numeric_limits<int>::max()) {
    num_bytes = std::numeric_limits<int>::max();
  }

  char* ptr = pbase();

  Status status = Status::OK();

  ORT_TRY {
    status = ToStatusAndRelease(out_stream_holder_.write_func(out_stream_holder_.stream_state,
                                                              ptr, num_bytes));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Caught exception while calling user's OrtOutStreamWriteFunc callback: ", e.what());
    });
  }

  if (!status.IsOK()) {
    last_status_ = std::move(status);
    return -1;
  }

  pbump(-static_cast<int>(num_bytes));  // Reset internal pointer to point to the beginning of the buffer_
  return 0;
}

}  // namespace epctx
}  // namespace onnxruntime

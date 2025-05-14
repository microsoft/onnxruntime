// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
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

  output_external_initializers_file_path = config_options.GetConfigOrDefault(
      kOrtSessionOptionsEpContextModelExternalInitializersFileName, "");
  output_external_initializer_size_threshold = 0;
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

// class OutStreamBuf

OutStreamBuf::OutStreamBuf(OutStreamHolder out_stream_holder) : out_stream_holder_(out_stream_holder) {
  setp(buffer_.data(), buffer_.data() + buffer_.size() - 1);  // Leave room for overflow character
}

OutStreamBuf::~OutStreamBuf() {
  sync();
}

std::streambuf::int_type OutStreamBuf::overflow(std::streambuf::int_type ch) {
  if (ch != traits_type::eof()) {
    *pptr() = static_cast<char>(ch);
    pbump(1);
  }

  if (FlushBuffer() == -1) {
    return traits_type::eof();
  }

  return ch;
}

int OutStreamBuf::sync() {
  return FlushBuffer();
}

int OutStreamBuf::FlushBuffer() {
  std::ptrdiff_t num_bytes = pptr() - pbase();
  if (num_bytes == 0) {
    return 0;
  }

  // Can only call pbump() with an int, so can only write at most 2^31 - 1.
  if (num_bytes > std::numeric_limits<int>::max()) {
    num_bytes = std::numeric_limits<int>::max();
  }

  std::ptrdiff_t bytes_remaining = num_bytes;
  char* ptr = pbase();

  while (bytes_remaining > 0) {
    size_t bytes_written = 0;
    Status status = Status::OK();

    ORT_TRY {
      status = ToStatus(out_stream_holder_.write_func(out_stream_holder_.stream_state,
                                                      ptr, bytes_remaining, &bytes_written));
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

    if (bytes_written > static_cast<size_t>(bytes_remaining)) {
      last_status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "OrtOutStreamWriteFunc wrote more bytes (", bytes_written,
                                     ") than requested (", bytes_remaining, ").");
      return -1;
    }

    bytes_remaining -= static_cast<std::ptrdiff_t>(bytes_written);
    ptr += bytes_written;
  }

  assert(ptr == pptr());
  pbump(-static_cast<int>(num_bytes));  // Reset internal pointer to point to the beginning of the buffer_
  return 0;
}

}  // namespace epctx
}  // namespace onnxruntime

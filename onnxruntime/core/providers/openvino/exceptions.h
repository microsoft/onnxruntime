// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <exception>
#include <regex>
#include <string>

#include "core/common/status.h"

namespace onnxruntime {
namespace openvino_ep {

struct ovep_exception : public std::exception {
  enum class type {
    compile_model,
    import_model,
    query_prop,
    read_model,
    unknown,
  };

  ovep_exception(const std::exception& ex, enum class type exception_type)
      : message_{ex.what()},
        type_{exception_type},
        error_code_{ze_result_code_from_string(message_)},
        error_name_{ze_result_name_from_string(message_)} {}

  ovep_exception(const std::string& message, enum class type exception_type)
      : message_{message},
        type_{exception_type},
        error_code_{ze_result_code_from_string(message)},
        error_name_{ze_result_name_from_string(message)} {}

  const char* what() const noexcept override {
    return message_.data();
  }

  uint32_t get_code() const { return error_code_; }

  operator common::Status() const {
    common::StatusCategory category_ort{common::ONNXRUNTIME};

    if (type_ == type::unknown) {
      return {category_ort, common::FAIL, message_};
    }

    // Newer drivers
    if ((type_ == type::import_model) &&
        (error_code_ == 0x7800000f /* ZE_RESULT_ERROR_INVALID_NATIVE_BINARY */)) {
      std::string message{error_name_ + ", code 0x" + std::to_string(error_code_) + "\nModel needs to be recompiled\n"};
      return {category_ort, common::INVALID_GRAPH, message};
    }

    std::string error_message = "Unhandled exception type: " + std::to_string(static_cast<int>(type_));
    return {category_ort, common::EP_FAIL, error_message};
  }

 protected:
  std::string message_;
  type type_{type::unknown};
  uint32_t error_code_{0};
  std::string error_name_;

 private:
  uint32_t ze_result_code_from_string(const std::string& ov_exception_string) {
    uint32_t error_code{0};
    std::regex error_code_pattern("code 0x([0-9a-fA-F]+)");
    std::smatch matches;
    if (std::regex_search(ov_exception_string, matches, error_code_pattern)) {
      std::from_chars(&(*matches[1].first), &(*matches[1].second), error_code, 16);
    }
    return error_code;
  }
  std::string ze_result_name_from_string(const std::string& ov_exception_string) {
    std::string error_message = "UNKNOWN NPU ERROR";
    std::regex error_message_pattern(R"(\bZE_\w*\b)");
    std::smatch matches;
    if (std::regex_search(ov_exception_string, matches, error_message_pattern)) {
      error_message = matches[0];
    }
    return error_message;
  }
};

}  // namespace openvino_ep
}  // namespace onnxruntime

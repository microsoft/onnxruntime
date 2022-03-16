// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Enable custom logging - this will cause TVM to use a custom implementation
// of tvm::runtime::detail::LogMessage. We use this to change the absolute
// file path to relative file path.

#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>


// TODO(agladyshev): Make conditional choice of sep for Windows and UNIX
std::string GetFileName(const std::string& file_path, char sep = '/') {
  return {std::next(file_path.begin(), file_path.find_last_of(sep) + 1),
          file_path.end()};
}

std::string GetTimedLogMessage(const std::string& file, int lineno, const std::string& message) {
  std::stringstream sstream;
  std::string file_name = GetFileName(file);
  std::time_t t = std::time(nullptr);
  sstream << "[" << std::put_time(std::localtime(&t), "%H:%M:%S") << "][TVM] "
          << file_name << ":" << lineno << ": " + message;
  return sstream.str();
}

namespace tvm {
namespace runtime {
namespace detail {
  void LogFatalImpl(const std::string& file, int lineno, const std::string& message) {
    throw std::runtime_error(GetTimedLogMessage(file, lineno, message));
  }

  void LogMessageImpl(const std::string& file, int lineno, const std::string& message) {
    std::cerr << GetTimedLogMessage(file, lineno, message) << std::endl;
  }

}  // namespace detail
}  // namespace runtime
}  // namespace tvm

#pragma once
#include <string>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace file_utils {

inline std::vector<char> ReadFile(const std::string& path) {
  if (!std::filesystem::exists(path)) {
    LOGS_DEFAULT(INFO) << "TensorRT RTX could not find the file and will create a new one " << path << std::endl;
    return {};
  }
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    ORT_THROW("Failed to open file: " + path);
  }
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(size);
  if (size > 0 && !file.read(buffer.data(), size)) {
    ORT_THROW("Failed to read file: " + path);
  }
  return buffer;
}

inline void WriteFile(const std::string& path, const void* data, size_t size) {
  if (std::filesystem::exists(path)) {
    std::ofstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
      ORT_THROW("Failed to open file for writing: " + path);
    }
    file.write(static_cast<const char*>(data), size);
  } else {
    LOGS_DEFAULT(INFO) << "TensorRT RTX a new file cache was written to " << path << std::endl;
    // Create new file
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file) {
      ORT_THROW("Failed to create file: " + path);
    }
    file.write(static_cast<const char*>(data), size);
  }
}

inline void WriteFile(const std::string& path, const std::vector<char>& data) { WriteFile(path, data.data(), data.size()); }

}  // namespace file_utils
}  // namespace onnxruntime
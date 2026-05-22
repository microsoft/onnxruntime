// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "plugin_ep_utils.h"

// Sample-only EPContext data helpers. These are intentionally outside the ORT C and EP ABI.
namespace ep_context_data_utils {

#ifdef _WIN32
inline std::wstring Utf8ToWideString(std::string_view value) {
  if (value.empty() || value.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return {};
  }

  const int wide_length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                              static_cast<int>(value.size()), nullptr, 0);
  if (wide_length <= 0) {
    return {};
  }

  std::wstring wide_value(static_cast<size_t>(wide_length), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()),
                      wide_value.data(), wide_length);
  return wide_value;
}

inline std::string WideToUtf8String(std::wstring_view value) {
  if (value.empty() || value.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return {};
  }

  const int utf8_length = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                                              nullptr, 0, nullptr, nullptr);
  if (utf8_length <= 0) {
    return {};
  }

  std::string utf8_value(static_cast<size_t>(utf8_length), '\0');
  WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), utf8_value.data(), utf8_length,
                      nullptr, nullptr);
  return utf8_value;
}
#endif

inline std::filesystem::path Utf8Path(const char* path) {
#ifdef _WIN32
  return std::filesystem::path{Utf8ToWideString(path)};
#else
  return std::filesystem::path{path};
#endif
}

inline std::string PathToUtf8String(const std::filesystem::path& path) {
#ifdef _WIN32
  return WideToUtf8String(path.wstring());
#else
  return path.string();
#endif
}

inline OrtStatus* ResolveEpContextDataPath(const OrtApi& api, const char* file_name, const OrtGraph* graph,
                                           std::filesystem::path& data_path) {
  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  data_path = Utf8Path(file_name);
  if (data_path.empty()) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name is not a valid path");
  }

  if (data_path.is_absolute() || graph == nullptr) {
    return nullptr;
  }

  const ORTCHAR_T* model_path = nullptr;
  RETURN_IF_ERROR(api.Graph_GetModelPath(graph, &model_path));
  if (model_path == nullptr || model_path[0] == 0) {
    return nullptr;
  }

  data_path = std::filesystem::path{model_path}.parent_path() / data_path;
  return nullptr;
}

inline OrtStatus* ReadEpContextDataFromFile(const OrtApi& api, const char* file_name, const OrtGraph* graph,
                                            std::vector<char>& data) {
  data.clear();

  std::filesystem::path data_path;
  RETURN_IF_ERROR(ResolveEpContextDataPath(api, file_name, graph, data_path));

  std::ifstream input_stream(data_path, std::ios::binary);
  if (!input_stream) {
    const std::string message = "Failed to open EPContext data file for read: " +
                                PathToUtf8String(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  data.assign(std::istreambuf_iterator<char>{input_stream}, std::istreambuf_iterator<char>{});
  if (input_stream.bad()) {
    const std::string message = "Failed to read EPContext data file: " +
                                PathToUtf8String(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  return nullptr;
}

inline OrtStatus* WriteEpContextDataToFile(const OrtApi& api, const char* file_name, const OrtGraph* graph,
                                           const void* buffer, size_t buffer_size) {
  if (buffer == nullptr && buffer_size != 0) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data buffer must not be null for non-empty data");
  }

  std::filesystem::path data_path;
  RETURN_IF_ERROR(ResolveEpContextDataPath(api, file_name, graph, data_path));

  std::ofstream output_stream(data_path, std::ios::binary);
  if (!output_stream) {
    const std::string message = "Failed to open EPContext data file for write: " +
                                PathToUtf8String(data_path);
    return api.CreateStatus(ORT_FAIL, message.c_str());
  }

  if (buffer_size != 0) {
    if (buffer_size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data buffer is too large to write");
    }

    output_stream.write(static_cast<const char*>(buffer), static_cast<std::streamsize>(buffer_size));
    if (!output_stream) {
      const std::string message = "Failed to write EPContext data file: " +
                                  PathToUtf8String(data_path);
      return api.CreateStatus(ORT_FAIL, message.c_str());
    }
  }

  return nullptr;
}

inline OrtStatus* ReadEpContextDataWithFileFallback(const OrtApi& api, const OrtEpApi& ep_api,
                                                    const OrtEpContextConfig* ep_context_config,
                                                    const char* file_name, const OrtGraph* graph,
                                                    std::vector<char>& data) {
  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  OrtReadEpContextDataFunc read_func = nullptr;
  void* read_state = nullptr;
  if (ep_context_config != nullptr) {
    RETURN_IF_ERROR(ep_api.EpContextConfig_GetEpContextDataReadFunc(ep_context_config, &read_func, &read_state));
  }

  if (read_func == nullptr) {
    return ReadEpContextDataFromFile(api, file_name, graph, data);
  }

  Ort::AllocatorWithDefaultOptions allocator;
  void* ep_context_data = nullptr;
  size_t ep_context_data_size = 0;
  OrtStatus* status = read_func(read_state, file_name, allocator, &ep_context_data, &ep_context_data_size);
  auto buffer_deleter = [&allocator](void* buffer_to_free) {
    if (buffer_to_free != nullptr) {
      allocator.Free(buffer_to_free);
    }
  };
  std::unique_ptr<void, decltype(buffer_deleter)> ep_context_data_guard(ep_context_data, buffer_deleter);

  if (status != nullptr) {
    return status;
  }

  if (ep_context_data_size != 0 && ep_context_data == nullptr) {
    return api.CreateStatus(
        ORT_FAIL, "OrtReadEpContextDataFunc returned a null buffer for non-empty EPContext data");
  }

  data.clear();
  if (ep_context_data != nullptr) {
    const char* ep_context_data_begin = static_cast<const char*>(ep_context_data);
    data.assign(ep_context_data_begin, ep_context_data_begin + ep_context_data_size);
  }

  return nullptr;
}

inline OrtStatus* WriteEpContextDataWithFileFallback(const OrtApi& api, const OrtEpApi& ep_api,
                                                     const OrtEpContextConfig* ep_context_config,
                                                     const char* file_name, const OrtGraph* graph,
                                                     const void* buffer, size_t buffer_size) {
  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  if (buffer == nullptr && buffer_size != 0) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data buffer must not be null for non-empty data");
  }

  OrtWriteEpContextDataFunc write_func = nullptr;
  void* write_state = nullptr;
  if (ep_context_config != nullptr) {
    RETURN_IF_ERROR(ep_api.EpContextConfig_GetEpContextDataWriteFunc(ep_context_config, &write_func, &write_state));
  }

  if (write_func != nullptr) {
    return write_func(write_state, file_name, buffer, buffer_size);
  }

  return WriteEpContextDataToFile(api, file_name, graph, buffer, buffer_size);
}

}  // namespace ep_context_data_utils

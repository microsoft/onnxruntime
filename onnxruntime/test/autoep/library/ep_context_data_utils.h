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
// Define NOMINMAX (and WIN32_LEAN_AND_MEAN) before <windows.h> so the min/max macros it would otherwise pull in do
// not clobber std::numeric_limits<...>::max() and std::min/std::max used in this header.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "plugin_ep_utils.h"
#include "onnxruntime_experimental_c_api.h"

// Sample-only EPContext data helpers shared by the example plugin EP and its tests. These are intentionally outside
// the ORT C and EP ABI and are provided as a reference for EP authors that need to handle external (non-embedded)
// EPContext binary data.
//
// The intended entry points for EP implementers are the ReadEpContextDataWithFileFallback /
// WriteEpContextDataWithFileFallback overloads: they prefer an application-supplied OrtReadNamedBufferFunc /
// OrtWriteNamedBufferFunc (carried by OrtEpContextConfig) and fall back to file I/O when no callback is configured.
// The other functions are lower-level building blocks. Production EPs should additionally apply their own sandboxing,
// size limits, and path policies; see the per-function notes on how untrusted, model-derived names are treated.
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
  if (path == nullptr || path[0] == '\0') {
    return {};
  }

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

inline bool ContainsPathTraversal(const std::filesystem::path& path) {
  const std::filesystem::path parent_dir{".."};
  for (const auto& component : path) {
    if (component == parent_dir) {
      return true;
    }
  }
  return false;
}

inline OrtStatus* ValidateEpContextDataName(const OrtApi& api, const char* file_name,
                                            std::filesystem::path& data_name) {
  data_name.clear();

  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  const auto candidate_path = Utf8Path(file_name);
  if (candidate_path.empty()) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name is not a valid path");
  }

  if (candidate_path.is_absolute()) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be absolute");
  }

  if (ContainsPathTraversal(candidate_path)) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not contain path traversal");
  }

  data_name = candidate_path;
  return nullptr;
}

// Resolve file_name (which can originate from an untrusted EPContext model's "ep_cache_context" attribute) as a
// model-relative path when graph is available. Trusted direct callers may pass an absolute path when graph is null.
// Production EPs should still apply their own sandboxing and size limits.
inline OrtStatus* ResolveEpContextDataPath(const OrtApi& api, const char* file_name, const OrtGraph* graph,
                                           std::filesystem::path& data_path) {
  data_path.clear();

  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  const auto candidate_path = Utf8Path(file_name);
  if (candidate_path.empty()) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name is not a valid path");
  }

  if (ContainsPathTraversal(candidate_path)) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not contain path traversal");
  }

  data_path = candidate_path;
  if (graph == nullptr) {
    return nullptr;
  }

  if (candidate_path.is_absolute()) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be absolute");
  }

  const ORTCHAR_T* model_path = nullptr;
  RETURN_IF_ERROR(api.Graph_GetModelPath(graph, &model_path));
  if (model_path == nullptr || model_path[0] == 0) {
    return nullptr;
  }

  data_path = std::filesystem::path{model_path}.parent_path() / data_path;
  return nullptr;
}

inline OrtStatus* ResolveEpContextDataOutputPath(const OrtApi& api, const char* file_name, const OrtGraph* graph,
                                                 std::filesystem::path& data_path) {
  data_path.clear();

  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  data_path = Utf8Path(file_name);
  if (data_path.empty()) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name is not a valid path");
  }

  if (ContainsPathTraversal(data_path)) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not contain path traversal");
  }

  // Trusted direct callers (graph == nullptr) may supply an absolute physical path. When a graph is present, the
  // name may be model-derived (untrusted), so reject absolute paths for symmetry with the read-side
  // ResolveEpContextDataPath instead of writing to an attacker-chosen absolute location.
  if (graph == nullptr) {
    return nullptr;
  }

  if (data_path.is_absolute()) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be absolute");
  }

  const ORTCHAR_T* model_path = nullptr;
  RETURN_IF_ERROR(api.Graph_GetModelPath(graph, &model_path));
  if (model_path == nullptr || model_path[0] == 0) {
    return nullptr;
  }

  data_path = std::filesystem::path{model_path}.parent_path() / data_path;
  return nullptr;
}

inline OrtStatus* WriteEpContextDataToResolvedFile(const OrtApi& api, const std::filesystem::path& data_path,
                                                   const void* buffer, size_t buffer_size) {
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
  return WriteEpContextDataToResolvedFile(api, data_path, buffer, buffer_size);
}

inline OrtStatus* ReadEpContextDataWithFileFallback(
    const OrtApi& api,
    OrtExperimental_OrtEpApi_EpContextConfig_GetEpContextDataReadFunc_SinceV28_Fn get_read_func,
    const OrtEpContextConfig* ep_context_config,
    const char* file_name, const OrtGraph* graph,
    std::vector<char>& data) {
  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
  if (ep_context_config != nullptr && get_read_func == nullptr) {
    return api.CreateStatus(ORT_NOT_IMPLEMENTED,
                            "OrtEpApi_EpContextConfig_GetEpContextDataReadFunc is not available");
  }
  if (ep_context_config != nullptr) {
    RETURN_IF_ERROR(get_read_func(ep_context_config, &read_func, &read_state));
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
        ORT_FAIL, "OrtReadNamedBufferFunc returned a null buffer for non-empty EPContext data");
  }

  data.clear();
  if (ep_context_data != nullptr) {
    const char* ep_context_data_begin = static_cast<const char*>(ep_context_data);
    data.assign(ep_context_data_begin, ep_context_data_begin + ep_context_data_size);
  }

  return nullptr;
}

// Reads EPContext binary data named `file_name`. If the session configured an OrtReadNamedBufferFunc (carried by
// `ep_context_config`), it is used; otherwise the data is read from a file. When `graph` is non-null it is the
// EPContext model graph: untrusted absolute/traversal names are rejected and relative names are resolved against the
// model directory. Pass `graph == nullptr` only for trusted callers supplying a physical path. `data` is cleared
// first and receives the bytes on success.
inline OrtStatus* ReadEpContextDataWithFileFallback(
    const OrtApi& api,
    const OrtEpContextConfig* ep_context_config,
    const char* file_name, const OrtGraph* graph,
    std::vector<char>& data) {
  return ReadEpContextDataWithFileFallback(
      api,
      Ort::Experimental::Get_OrtEpApi_EpContextConfig_GetEpContextDataReadFunc_SinceV28_Fn(&api),
      ep_context_config, file_name, graph, data);
}

inline OrtStatus* WriteEpContextDataWithFileFallback(
    const OrtApi& api,
    OrtExperimental_OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28_Fn get_write_func,
    const OrtEpContextConfig* ep_context_config,
    const char* file_name, const char* fallback_file_name,
    const OrtGraph* graph,
    const void* buffer, size_t buffer_size) {
  if (file_name == nullptr || file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  }

  if (buffer == nullptr && buffer_size != 0) {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data buffer must not be null for non-empty data");
  }

  OrtWriteNamedBufferFunc write_func = nullptr;
  void* write_state = nullptr;
  if (ep_context_config != nullptr && get_write_func == nullptr) {
    return api.CreateStatus(ORT_NOT_IMPLEMENTED,
                            "OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc is not available");
  }
  if (ep_context_config != nullptr) {
    RETURN_IF_ERROR(get_write_func(ep_context_config, &write_func, &write_state));
  }

  // The app-supplied write callback owns its own logical namespace, so file_name is passed through unmodified.
  // Only the file-fallback path below maps a name onto the filesystem, so it validates the logical name there.
  if (write_func != nullptr) {
    return write_func(write_state, file_name, buffer, buffer_size);
  }

  std::filesystem::path logical_path;
  RETURN_IF_ERROR(ValidateEpContextDataName(api, file_name, logical_path));

  if (fallback_file_name == nullptr || fallback_file_name[0] == '\0') {
    return api.CreateStatus(ORT_INVALID_ARGUMENT, "EPContext data fallback file name must not be empty");
  }

  std::filesystem::path data_path;
  RETURN_IF_ERROR(ResolveEpContextDataOutputPath(api, fallback_file_name, graph, data_path));
  return WriteEpContextDataToResolvedFile(api, data_path, buffer, buffer_size);
}

// Writes EPContext binary data. If the compilation configured an OrtWriteNamedBufferFunc (carried by
// `ep_context_config`), it is used and `file_name` is passed through unmodified as the logical name. Otherwise the
// data is written to a file at `fallback_file_name`, which is resolved against the model directory when `graph` is
// non-null (and rejected if absolute in that case). `graph == nullptr` denotes a trusted caller that may supply an
// absolute physical path. `buffer` may be null only when `buffer_size` is 0.
inline OrtStatus* WriteEpContextDataWithFileFallback(
    const OrtApi& api,
    const OrtEpContextConfig* ep_context_config,
    const char* file_name, const char* fallback_file_name,
    const OrtGraph* graph,
    const void* buffer, size_t buffer_size) {
  return WriteEpContextDataWithFileFallback(
      api,
      Ort::Experimental::Get_OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28_Fn(&api),
      ep_context_config, file_name, fallback_file_name, graph, buffer, buffer_size);
}

inline OrtStatus* WriteEpContextDataWithFileFallback(
    const OrtApi& api,
    OrtExperimental_OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28_Fn get_write_func,
    const OrtEpContextConfig* ep_context_config,
    const char* file_name, const OrtGraph* graph,
    const void* buffer, size_t buffer_size) {
  return WriteEpContextDataWithFileFallback(api, get_write_func, ep_context_config, file_name, file_name, graph, buffer,
                                            buffer_size);
}

// Convenience overload that uses `file_name` as both the logical callback name and the file-fallback path.
inline OrtStatus* WriteEpContextDataWithFileFallback(
    const OrtApi& api,
    const OrtEpContextConfig* ep_context_config,
    const char* file_name, const OrtGraph* graph,
    const void* buffer, size_t buffer_size) {
  return WriteEpContextDataWithFileFallback(api, ep_context_config, file_name, file_name, graph, buffer, buffer_size);
}

}  // namespace ep_context_data_utils

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#define RETURN_IF_ERROR(fn)    \
  do {                         \
    OrtStatus* _status = (fn); \
    if (_status != nullptr) {  \
      return _status;          \
    }                          \
  } while (0)

#define RETURN_IF(cond, ort_api, msg)                    \
  do {                                                   \
    if ((cond)) {                                        \
      return (ort_api).CreateStatus(ORT_EP_FAIL, (msg)); \
    }                                                    \
  } while (0)

// see ORT_ENFORCE for implementations that also capture a stack trace and work in builds with exceptions disabled
// NOTE: In this simplistic implementation you must provide an argument, even it if's an empty string
#define EP_ENFORCE(condition, ...)                       \
  do {                                                   \
    if (!(condition)) {                                  \
      std::ostringstream oss;                            \
      oss << "EP_ENFORCE failed: " << #condition << " "; \
      oss << __VA_ARGS__;                                \
      throw std::runtime_error(oss.str());               \
    }                                                    \
  } while (false)

#ifdef _WIN32
#define EP_WSTR(x) L##x
#define EP_FILE_INTERNAL(x) EP_WSTR(x)
#define EP_FILE EP_FILE_INTERNAL(__FILE__)
#else
#define EP_FILE __FILE__
#endif

#define LOG(level, ...)                                                                                             \
  do {                                                                                                              \
    std::ostringstream ss;                                                                                          \
    ss << __VA_ARGS__;                                                                                              \
    api_.Logger_LogMessage(&logger_, ORT_LOGGING_LEVEL_##level, ss.str().c_str(), EP_FILE, __LINE__, __FUNCTION__); \
  } while (false)

#define RETURN_ERROR(code, ...)                       \
  do {                                                \
    std::ostringstream ss;                            \
    ss << __VA_ARGS__;                                \
    return api_.CreateStatus(code, ss.str().c_str()); \
  } while (false)

#define THROW(...)       \
  std::ostringstream ss; \
  ss << __VA_ARGS__;     \
  throw std::runtime_error(ss.str())

struct ApiPtrs {
  const OrtApi& ort_api;
  const OrtEpApi& ep_api;
  const OrtModelEditorApi& model_editor_api;
};

using AllocatorUniquePtr = std::unique_ptr<OrtAllocator, std::function<void(OrtAllocator*)>>;

// Helper to release Ort one or more objects obtained from the public C API at the end of their scope.
template <typename T>
struct DeferOrtRelease {
  DeferOrtRelease(T** object_ptr, std::function<void(T*)> release_func)
      : objects_(object_ptr), count_(1), release_func_(release_func) {}

  DeferOrtRelease(T** objects, size_t count, std::function<void(T*)> release_func)
      : objects_(objects), count_(count), release_func_(release_func) {}

  ~DeferOrtRelease() {
    if (objects_ != nullptr && count_ > 0) {
      for (size_t i = 0; i < count_; ++i) {
        if (objects_[i] != nullptr) {
          release_func_(objects_[i]);
          objects_[i] = nullptr;
        }
      }
    }
  }
  T** objects_ = nullptr;
  size_t count_ = 0;
  std::function<void(T*)> release_func_ = nullptr;
};

struct FloatInitializer {
  std::vector<int64_t> shape;
  std::vector<float> data;
};

// Returns an entry in the session option configurations, or a default value if not present.
OrtStatus* GetSessionConfigEntryOrDefault(const OrtApi& ort_api, const OrtSessionOptions& session_options,
                                          const char* config_key, const std::string& default_val,
                                          /*out*/ std::string& config_val);

// Returns true (via output parameter) if the given OrtValueInfo represents a float tensor.
OrtStatus* IsFloatTensor(const OrtApi& ort_api, const OrtValueInfo* value_info, bool& result);

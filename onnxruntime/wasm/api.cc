// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "api.h"

#include "core/session/onnxruntime_cxx_api.h"

#include <iostream>
#include <vector>

namespace {
OrtEnv* g_env;
}  // namespace

OrtErrorCode CheckStatus(OrtStatusPtr status) {
  OrtErrorCode error_code = ORT_OK;
  if (status) {
    std::string error_message = Ort::GetApi().GetErrorMessage(status);
    error_code = Ort::GetApi().GetErrorCode(status);
    std::cerr << Ort::Exception(std::move(error_message), error_code).what() << std::endl;
    Ort::GetApi().ReleaseStatus(status);
  }
  return error_code;
}

#define CHECK_STATUS(ORT_API_NAME, ...) \
  CheckStatus(Ort::GetApi().ORT_API_NAME(__VA_ARGS__))

#define RETURN_ERROR_CODE_IF_ERROR(ORT_API_NAME, ...)         \
  do {                                                        \
    int error_code = CHECK_STATUS(ORT_API_NAME, __VA_ARGS__); \
    if (error_code != ORT_OK) {                               \
      return error_code;                                      \
    }                                                         \
  } while (false)

// TODO: This macro can be removed when we changed all APIs to return a status code.
#define RETURN_NULLPTR_IF_ERROR(ORT_API_NAME, ...)           \
  do {                                                       \
    if (CHECK_STATUS(ORT_API_NAME, __VA_ARGS__) != ORT_OK) { \
      return nullptr;                                        \
    }                                                        \
  } while (false)

int OrtInit(int num_threads, int logging_level) {
  // Assume that a logging level is check and properly set at JavaScript
#if defined(__EMSCRIPTEN_PTHREADS__)
  OrtThreadingOptions* tp_options = nullptr;
  RETURN_ERROR_CODE_IF_ERROR(CreateThreadingOptions, &tp_options);
  RETURN_ERROR_CODE_IF_ERROR(SetGlobalIntraOpNumThreads, tp_options, num_threads);
  RETURN_ERROR_CODE_IF_ERROR(SetGlobalInterOpNumThreads, tp_options, 1);

  return CHECK_STATUS(CreateEnvWithGlobalThreadPools,
                      static_cast<OrtLoggingLevel>(logging_level),
                      "Default",
                      tp_options,
                      &g_env);
#else
  return CHECK_STATUS(CreateEnv, static_cast<OrtLoggingLevel>(logging_level), "Default", &g_env);
#endif
}

OrtSessionOptions* OrtCreateSessionOptions(size_t graph_optimization_level,
                                           bool enable_cpu_mem_arena,
                                           bool enable_mem_pattern,
                                           size_t execution_mode,
                                           bool /* enable_profiling */,
                                           const char* /* profile_file_prefix */,
                                           const char* log_id,
                                           size_t log_severity_level,
                                           size_t log_verbosity_level) {
  OrtSessionOptions* session_options = nullptr;
  RETURN_NULLPTR_IF_ERROR(CreateSessionOptions, &session_options);

  // assume that a graph optimization level is checked and properly set at JavaScript
  RETURN_NULLPTR_IF_ERROR(SetSessionGraphOptimizationLevel,
                          session_options,
                          static_cast<GraphOptimizationLevel>(graph_optimization_level));

  if (enable_cpu_mem_arena) {
    RETURN_NULLPTR_IF_ERROR(EnableCpuMemArena, session_options);
  } else {
    RETURN_NULLPTR_IF_ERROR(DisableCpuMemArena, session_options);
  }

  if (enable_mem_pattern) {
    RETURN_NULLPTR_IF_ERROR(EnableCpuMemArena, session_options);
  } else {
    RETURN_NULLPTR_IF_ERROR(DisableCpuMemArena, session_options);
  }

  // assume that an execution mode is checked and properly set at JavaScript
  RETURN_NULLPTR_IF_ERROR(SetSessionExecutionMode, session_options, static_cast<ExecutionMode>(execution_mode));

  // TODO: support profling

  if (log_id != nullptr) {
    RETURN_NULLPTR_IF_ERROR(SetSessionLogId, session_options, log_id);
  }

  // assume that a log severity level is checked and properly set at JavaScript
  RETURN_NULLPTR_IF_ERROR(SetSessionLogSeverityLevel, session_options, log_severity_level);

  RETURN_NULLPTR_IF_ERROR(SetSessionLogVerbosityLevel, session_options, log_verbosity_level);

  return session_options;
}

int OrtAddSessionConfigEntry(OrtSessionOptions* session_options,
                             const char* config_key,
                             const char* config_value) {
  return CHECK_STATUS(AddSessionConfigEntry, session_options, config_key, config_value);
}

void OrtReleaseSessionOptions(OrtSessionOptions* session_options) {
  Ort::GetApi().ReleaseSessionOptions(session_options);
}

OrtSession* OrtCreateSession(void* data, size_t data_length, OrtSessionOptions* session_options) {
  // OrtSessionOptions must not be nullptr.
  if (session_options == nullptr) {
    return nullptr;
  }

#ifdef ENABLE_EXTENSION_CUSTOM_OPS
  // Enable ORT CustomOps in onnxruntime-extensions
  RETURN_NULLPTR_IF_ERROR(EnableOrtCustomOps, session_options);
#endif

#if defined(__EMSCRIPTEN_PTHREADS__)
  RETURN_NULLPTR_IF_ERROR(DisablePerSessionThreads, session_options);
#else
  // must disable thread pool when WebAssembly multi-threads support is disabled.
  RETURN_NULLPTR_IF_ERROR(SetIntraOpNumThreads, session_options, 1);
  RETURN_NULLPTR_IF_ERROR(SetSessionExecutionMode, session_options, ORT_SEQUENTIAL);
#endif

  OrtSession* session = nullptr;
  return (CHECK_STATUS(CreateSessionFromArray, g_env, data, data_length, session_options, &session) == ORT_OK)
             ? session
             : nullptr;
}

void OrtReleaseSession(OrtSession* session) {
  Ort::GetApi().ReleaseSession(session);
}

size_t OrtGetInputCount(OrtSession* session) {
  size_t input_count = 0;
  return (CHECK_STATUS(SessionGetInputCount, session, &input_count) == ORT_OK) ? input_count : 0;
}

size_t OrtGetOutputCount(OrtSession* session) {
  size_t output_count = 0;
  return (CHECK_STATUS(SessionGetOutputCount, session, &output_count) == ORT_OK) ? output_count : 0;
}

char* OrtGetInputName(OrtSession* session, size_t index) {
  OrtAllocator* allocator = nullptr;
  RETURN_NULLPTR_IF_ERROR(GetAllocatorWithDefaultOptions, &allocator);

  char* input_name = nullptr;
  return (CHECK_STATUS(SessionGetInputName, session, index, allocator, &input_name) == ORT_OK)
             ? input_name
             : nullptr;
}

char* OrtGetOutputName(OrtSession* session, size_t index) {
  OrtAllocator* allocator = nullptr;
  RETURN_NULLPTR_IF_ERROR(GetAllocatorWithDefaultOptions, &allocator);

  char* output_name = nullptr;
  return (CHECK_STATUS(SessionGetOutputName, session, index, allocator, &output_name) == ORT_OK)
             ? output_name
             : nullptr;
}

void OrtFree(void* ptr) {
  OrtAllocator* allocator = nullptr;
  if (CHECK_STATUS(GetAllocatorWithDefaultOptions, &allocator) == ORT_OK) {
    allocator->Free(allocator, ptr);
  }
}

OrtValue* OrtCreateTensor(int data_type, void* data, size_t data_length, size_t* dims, size_t dims_length) {
  std::vector<int64_t> shapes(dims_length);
  for (size_t i = 0; i < dims_length; i++) {
    shapes[i] = dims[i];
  }

  if (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    OrtAllocator* allocator = nullptr;
    RETURN_NULLPTR_IF_ERROR(GetAllocatorWithDefaultOptions, &allocator);

    OrtValue* value = nullptr;
    RETURN_NULLPTR_IF_ERROR(CreateTensorAsOrtValue, allocator,
                            dims_length > 0 ? shapes.data() : nullptr, dims_length,
                            ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &value);

    const char* const* strings = reinterpret_cast<const char* const*>(data);
    RETURN_NULLPTR_IF_ERROR(FillStringTensor, value, strings, data_length / sizeof(const char*));

    return value;
  } else {
    OrtMemoryInfo* memoryInfo = nullptr;
    RETURN_NULLPTR_IF_ERROR(CreateCpuMemoryInfo, OrtDeviceAllocator, OrtMemTypeDefault, &memoryInfo);

    OrtValue* value = nullptr;
    int error_code = CHECK_STATUS(CreateTensorWithDataAsOrtValue, memoryInfo, data, data_length,
                                  dims_length > 0 ? shapes.data() : nullptr, dims_length,
                                  static_cast<ONNXTensorElementDataType>(data_type), &value);

    Ort::GetApi().ReleaseMemoryInfo(memoryInfo);
    return (error_code == ORT_OK) ? value : nullptr;
  }
}

int OrtGetTensorData(OrtValue* tensor, int* data_type, void** data, size_t** dims, size_t* dims_length) {
#define RELEASE_AND_RETURN_ERROR_CODE_IF_ERROR(ORT_API_NAME, ...) \
  do {                                                            \
    int error_code = CHECK_STATUS(ORT_API_NAME, __VA_ARGS__);     \
    if (error_code != ORT_OK) {                                   \
      if (info != nullptr) {                                      \
        Ort::GetApi().ReleaseTensorTypeAndShapeInfo(info);        \
      }                                                           \
      if (allocator != nullptr && p_dims != nullptr) {            \
        allocator->Free(allocator, p_dims);                       \
      }                                                           \
      if (allocator != nullptr && p_string_data != nullptr) {     \
        allocator->Free(allocator, p_string_data);                \
      }                                                           \
      return error_code;                                          \
    }                                                             \
  } while (false)

  OrtTensorTypeAndShapeInfo* info = nullptr;
  OrtAllocator* allocator = nullptr;
  size_t* p_dims = nullptr;
  void* p_string_data = nullptr;

  RETURN_ERROR_CODE_IF_ERROR(GetTensorTypeAndShape, tensor, &info);

  size_t dims_len = 0;
  RELEASE_AND_RETURN_ERROR_CODE_IF_ERROR(GetDimensionsCount, info, &dims_len);

  RELEASE_AND_RETURN_ERROR_CODE_IF_ERROR(GetAllocatorWithDefaultOptions, &allocator);
  p_dims = reinterpret_cast<size_t*>(allocator->Alloc(allocator, sizeof(size_t) * dims_len));

  RELEASE_AND_RETURN_ERROR_CODE_IF_ERROR(GetTensorMutableData, tensor, data);

  ONNXTensorElementDataType type;
  RELEASE_AND_RETURN_ERROR_CODE_IF_ERROR(GetTensorElementType, info, &type);
  *data_type = static_cast<int>(type);

  *dims_length = dims_len;
  std::vector<int64_t> shape(dims_len, 0);
  RELEASE_AND_RETURN_ERROR_CODE_IF_ERROR(GetDimensions, info, shape.data(), shape.size());
  for (size_t i = 0; i < dims_len; i++) {
    p_dims[i] = static_cast<size_t>(shape[i]);
  }
  *dims = p_dims;

  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    size_t num_elements;
    RELEASE_AND_RETURN_ERROR_CODE_IF_ERROR(GetTensorShapeElementCount, info, &num_elements);

    // NOTE: ORT C-API does not expose an interface for users to get string raw data directly. There is always a copy.
    //       we can use the tensor raw data because it is type of "std::string *", which is very starightforward to
    //       implement and can also save memory usage. However, this approach depends on the Tensor's implementation
    //       details. So we have to copy the string content here.

    size_t string_data_length;
    RELEASE_AND_RETURN_ERROR_CODE_IF_ERROR(GetStringTensorDataLength, tensor, &string_data_length);

    // The buffer contains following data:
    //  - a sequence of pointers to (const char*), size = num_elements * sizeof(const char*).
    //  - followed by a raw buffer to store string content, size = string_data_length + 1.
    static_assert(sizeof(const char*) == sizeof(size_t), "size of a pointer and a size_t value should be the same.");

    size_t string_data_offset = num_elements * sizeof(const char*);
    size_t buf_size = string_data_offset + string_data_length;
    p_string_data = allocator->Alloc(allocator, buf_size + 1);
    void* p_string_content = reinterpret_cast<char*>(p_string_data) + string_data_offset;

    size_t* p_offsets = reinterpret_cast<size_t*>(p_string_data);
    RELEASE_AND_RETURN_ERROR_CODE_IF_ERROR(GetStringTensorContent, tensor, p_string_content, string_data_length, p_offsets, num_elements);

    // replace offsets by pointers
    const char** p_c_strs = reinterpret_cast<const char**>(p_offsets);
    for (size_t i = 0; i < num_elements; i++) {
      p_c_strs[i] = reinterpret_cast<const char*>(p_string_content) + p_offsets[i];
    }

    // put null at the last char
    reinterpret_cast<char*>(p_string_data)[buf_size] = '\0';
    *data = p_string_data;
  }

  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(info);
  return ORT_OK;
}

void OrtReleaseTensor(OrtValue* tensor) {
  Ort::GetApi().ReleaseValue(tensor);
}

OrtRunOptions* OrtCreateRunOptions(size_t log_severity_level,
                                   size_t log_verbosity_level,
                                   bool terminate,
                                   const char* tag) {
  OrtRunOptions* run_options = nullptr;
  RETURN_NULLPTR_IF_ERROR(CreateRunOptions, &run_options);

  // Assume that a logging level is check and properly set at JavaScript
  RETURN_NULLPTR_IF_ERROR(RunOptionsSetRunLogSeverityLevel, run_options, log_severity_level);

  RETURN_NULLPTR_IF_ERROR(RunOptionsSetRunLogVerbosityLevel, run_options, log_verbosity_level);

  if (terminate) {
    RETURN_NULLPTR_IF_ERROR(RunOptionsSetTerminate, run_options);
  } else {
    RETURN_NULLPTR_IF_ERROR(RunOptionsUnsetTerminate, run_options);
  }

  if (tag != nullptr) {
    RETURN_NULLPTR_IF_ERROR(RunOptionsSetRunTag, run_options, tag);
  }

  return run_options;
}

int OrtAddRunConfigEntry(OrtRunOptions* run_options,
                         const char* config_key,
                         const char* config_value) {
  return CHECK_STATUS(AddRunConfigEntry, run_options, config_key, config_value);
}

void OrtReleaseRunOptions(OrtRunOptions* run_options) {
  Ort::GetApi().ReleaseRunOptions(run_options);
}

int OrtRun(OrtSession* session,
           const char** input_names, const ort_tensor_handle_t* inputs, size_t input_count,
           const char** output_names, size_t output_count, ort_tensor_handle_t* outputs,
           OrtRunOptions* run_options) {
  return CHECK_STATUS(Run, session, run_options, input_names, inputs, input_count, output_names, output_count, outputs);
}

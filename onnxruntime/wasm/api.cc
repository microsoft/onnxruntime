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

#define RETURN_ERROR_CODE_IF_ERROR(ORT_API_NAME, ...)                       \
  do {                                                                      \
    int error_code = CHECK_STATUS(ORT_API_NAME, __VA_ARGS__);               \
    if (error_code != ORT_OK) {                                             \
      return error_code;                                                    \
    }                                                                       \
  } while (false)

// TODO: This macro can be removed when we changed all APIs to return a status code.
#define RETURN_NULLPTR_IF_ERROR(ORT_API_NAME, ...)                          \
  do {                                                                      \
    if (CHECK_STATUS(ORT_API_NAME, __VA_ARGS__) != ORT_OK) {                \
      return nullptr;                                                       \
    }                                                                       \
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

OrtSessionOptions* OrtCreateSessionOptions() {
  OrtSessionOptions* session_options = nullptr;
  return (CHECK_STATUS(CreateSessionOptions, &session_options) == ORT_OK) ? session_options : nullptr;
}

void OrtReleaseSessionOptions(OrtSessionOptions* session_options) {
  Ort::GetApi().ReleaseSessionOptions(session_options);
}

int OrtSetSessionGraphOptimizationLevel(OrtSessionOptions* session_options, size_t level) {
  // Assume that a graph optimization level is check and properly set at JavaScript
  return CHECK_STATUS(SetSessionGraphOptimizationLevel, session_options, static_cast<GraphOptimizationLevel>(level));
}

int OrtEnableCpuMemArena(OrtSessionOptions* session_options) {
  return CHECK_STATUS(EnableCpuMemArena, session_options);
}

int OrtDisableCpuMemArena(OrtSessionOptions* session_options) {
  return CHECK_STATUS(DisableCpuMemArena, session_options);
}

int OrtEnableMemPattern(OrtSessionOptions* session_options) {
  return CHECK_STATUS(EnableMemPattern, session_options);
}

int OrtDisableMemPattern(OrtSessionOptions* session_options) {
  return CHECK_STATUS(DisableMemPattern, session_options);
}

int OrtSetSessionExecutionMode(OrtSessionOptions* session_options, size_t mode) {
  // Assume that an execution mode is check and properly set at JavaScript
  return CHECK_STATUS(SetSessionExecutionMode, session_options, static_cast<ExecutionMode>(mode));
}

int OrtSetSessionLogId(OrtSessionOptions* session_options, const char* logid) {
  return CHECK_STATUS(SetSessionLogId, session_options, logid);
}

int OrtSetSessionLogSeverityLevel(OrtSessionOptions* session_options, size_t level) {
  return CHECK_STATUS(SetSessionLogSeverityLevel, session_options, level);
}

OrtSession* OrtCreateSession(void* data, size_t data_length, OrtSessionOptions* session_options) {
  // OrtSessionOptions must not be nullptr.
  if (session_options == nullptr) {
    return nullptr;
  }

#if defined(__EMSCRIPTEN_PTHREADS__)
  RETURN_NULLPTR_IF_ERROR(DisablePerSessionThreads, session_options);
#else
  // must disable thread pool when WebAssembly multi-threads support is disabled.
  RETURN_NULLPTR_IF_ERROR(SetIntraOpNumThreads, session_options, 1);
  RETURN_NULLPTR_IF_ERROR(SetSessionExecutionMode, session_options, ORT_SEQUENTIAL);
#endif

  OrtSession* session = nullptr;
  return (CHECK_STATUS(CreateSessionFromArray, g_env, data, data_length, session_options, &session) == ORT_OK)
             ? session : nullptr;
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
             ? input_name : nullptr;
}

char* OrtGetOutputName(OrtSession* session, size_t index) {
  OrtAllocator* allocator = nullptr;
  RETURN_NULLPTR_IF_ERROR(GetAllocatorWithDefaultOptions, &allocator);

  char* output_name = nullptr;
  return (CHECK_STATUS(SessionGetOutputName, session, index, allocator, &output_name) == ORT_OK)
             ? output_name : nullptr;
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

  OrtMemoryInfo* memoryInfo = nullptr;
  RETURN_NULLPTR_IF_ERROR(CreateCpuMemoryInfo, OrtDeviceAllocator, OrtMemTypeDefault, &memoryInfo);

  OrtValue* value = nullptr;
  int error_code = CHECK_STATUS(CreateTensorWithDataAsOrtValue, memoryInfo, data, data_length,
              dims_length > 0 ? shapes.data() : nullptr, dims_length,
              static_cast<ONNXTensorElementDataType>(data_type), &value);

  Ort::GetApi().ReleaseMemoryInfo(memoryInfo);
  return (error_code == ORT_OK) ? value : nullptr;
}

int OrtGetTensorData(OrtValue* tensor, int* data_type, void** data, size_t** dims, size_t* dims_length) {
#define RELEASE_AND_RETURN_ERROR_CODE_IF_ERROR(ORT_API_NAME, ...)           \
  do {                                                                      \
    int error_code = CHECK_STATUS(ORT_API_NAME, __VA_ARGS__);               \
    if (error_code != ORT_OK) {                                             \
      if (info != nullptr) {                                                \
        Ort::GetApi().ReleaseTensorTypeAndShapeInfo(info);                  \
      }                                                                     \
      if (allocator != nullptr && p_dims != nullptr) {                      \
        allocator->Free(allocator, p_dims);                                 \
      }                                                                     \
      return error_code;                                                    \
    }                                                                       \
  } while (false)

  OrtTensorTypeAndShapeInfo* info = nullptr;
  OrtAllocator* allocator = nullptr;
  size_t* p_dims = nullptr;

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

  Ort::GetApi().ReleaseTensorTypeAndShapeInfo(info);
  return ORT_OK;
}

void OrtReleaseTensor(OrtValue* tensor) {
  Ort::GetApi().ReleaseValue(tensor);
}

OrtRunOptions* OrtCreateRunOptions() {
  OrtRunOptions* run_options = nullptr;
  return (CHECK_STATUS(CreateRunOptions, &run_options) == ORT_OK) ? run_options : nullptr;
}

void OrtReleaseRunOptions(OrtRunOptions* run_options) {
  Ort::GetApi().ReleaseRunOptions(run_options);
}

int OrtRunOptionsSetRunLogSeverityLevel(OrtRunOptions* run_options, size_t level) {
  return CHECK_STATUS(RunOptionsSetRunLogSeverityLevel, run_options, level);
}

int OrtRunOptionsSetRunTag(OrtRunOptions* run_options, const char* tag) {
  return CHECK_STATUS(RunOptionsSetRunTag, run_options, tag);
}

int OrtRun(OrtSession* session,
           const char** input_names, const ort_tensor_handle_t* inputs, size_t input_count,
           const char** output_names, size_t output_count, ort_tensor_handle_t* outputs,
           OrtRunOptions* run_options) {
  return CHECK_STATUS(Run, session, run_options, input_names, inputs, input_count, output_names, output_count, outputs);
}

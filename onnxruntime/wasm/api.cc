// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "api.h"

#include "core/session/onnxruntime_cxx_api.h"

#include <iostream>
#include <vector>

namespace {
OrtEnv* g_env;
OrtErrorCode g_last_error_code;
std::string g_last_error_message;
}  // namespace

static_assert(sizeof(const char*) == sizeof(size_t), "size of a pointer and a size_t value should be the same.");
static_assert(sizeof(size_t) == 4, "size of size_t should be 4 in this build (wasm32).");

OrtErrorCode CheckStatus(OrtStatusPtr status) {
  if (status) {
    std::string error_message = Ort::GetApi().GetErrorMessage(status);
    g_last_error_code = Ort::GetApi().GetErrorCode(status);
    g_last_error_message = Ort::Exception(std::move(error_message), g_last_error_code).what();
    Ort::GetApi().ReleaseStatus(status);
  } else {
    g_last_error_code = ORT_OK;
    g_last_error_message.clear();
  }
  return g_last_error_code;
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

#define RETURN_NULLPTR_IF_ERROR(ORT_API_NAME, ...)           \
  do {                                                       \
    if (CHECK_STATUS(ORT_API_NAME, __VA_ARGS__) != ORT_OK) { \
      return nullptr;                                        \
    }                                                        \
  } while (false)

// use auto release macros to make sure resources get released on function return.

// create a unique_ptr wrapper for auto release
#define REGISTER_AUTO_RELEASE(T, var, release_t, release_func) \
  std::unique_ptr<T, release_t> auto_release_##var { var, release_func }
// register auto release for handle of Ort API resources
#define REGISTER_AUTO_RELEASE_HANDLE(T, var) \
  REGISTER_AUTO_RELEASE(Ort##T, var, void (*)(Ort##T*), [](Ort##T* p) { Ort::GetApi().Release##T(p); })
// register auto release for Ort allocated buffers
#define REGISTER_AUTO_RELEASE_BUFFER(T, var, allocator)                                     \
  auto auto_release_##var##_deleter = [allocator](T* p) { allocator->Free(allocator, p); }; \
  REGISTER_AUTO_RELEASE(T, var, decltype(auto_release_##var##_deleter), auto_release_##var##_deleter)
// unregister the auto release wrapper
#define UNREGISTER_AUTO_RELEASE(var) auto_release_##var.release()

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

void OrtGetLastError(int* error_code, const char** error_message) {
  *error_code = g_last_error_code;
  *error_message = g_last_error_message.empty() ? nullptr : g_last_error_message.c_str();
}

OrtSessionOptions* OrtCreateSessionOptions(size_t graph_optimization_level,
                                           bool enable_cpu_mem_arena,
                                           bool enable_mem_pattern,
                                           size_t execution_mode,
                                           bool enable_profiling,
                                           const char* /*profile_file_prefix*/,
                                           const char* log_id,
                                           size_t log_severity_level,
                                           size_t log_verbosity_level,
                                           const char* optimized_model_filepath) {
  OrtSessionOptions* session_options = nullptr;
  RETURN_NULLPTR_IF_ERROR(CreateSessionOptions, &session_options);
  REGISTER_AUTO_RELEASE_HANDLE(SessionOptions, session_options);

  if (optimized_model_filepath) {
    RETURN_NULLPTR_IF_ERROR(SetOptimizedModelFilePath, session_options, optimized_model_filepath);
  }

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
    RETURN_NULLPTR_IF_ERROR(EnableMemPattern, session_options);
  } else {
    RETURN_NULLPTR_IF_ERROR(DisableMemPattern, session_options);
  }

  // assume that an execution mode is checked and properly set at JavaScript
  RETURN_NULLPTR_IF_ERROR(SetSessionExecutionMode, session_options, static_cast<ExecutionMode>(execution_mode));

  // TODO: support profling
  if (enable_profiling) {
    RETURN_NULLPTR_IF_ERROR(EnableProfiling, session_options, "");
  } else {
    RETURN_NULLPTR_IF_ERROR(DisableProfiling, session_options);
  }

  if (log_id != nullptr) {
    RETURN_NULLPTR_IF_ERROR(SetSessionLogId, session_options, log_id);
  }

  // assume that a log severity level is checked and properly set at JavaScript
  RETURN_NULLPTR_IF_ERROR(SetSessionLogSeverityLevel, session_options, log_severity_level);

  RETURN_NULLPTR_IF_ERROR(SetSessionLogVerbosityLevel, session_options, log_verbosity_level);

#ifdef ENABLE_EXTENSION_CUSTOM_OPS
  // Enable ORT CustomOps in onnxruntime-extensions
  RETURN_NULLPTR_IF_ERROR(EnableOrtCustomOps, session_options);
#endif

  return UNREGISTER_AUTO_RELEASE(session_options);
}

int OrtAppendExecutionProvider(ort_session_options_handle_t session_options, const char* name) {
  return CHECK_STATUS(SessionOptionsAppendExecutionProvider, session_options, name, nullptr, nullptr, 0);
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

int OrtGetInputOutputCount(OrtSession* session, size_t* input_count, size_t* output_count) {
  RETURN_ERROR_CODE_IF_ERROR(SessionGetInputCount, session, input_count);
  RETURN_ERROR_CODE_IF_ERROR(SessionGetOutputCount, session, output_count);
  return ORT_OK;
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
    REGISTER_AUTO_RELEASE_HANDLE(Value, value);

    const char* const* strings = reinterpret_cast<const char* const*>(data);
    RETURN_NULLPTR_IF_ERROR(FillStringTensor, value, strings, data_length / sizeof(const char*));

    return UNREGISTER_AUTO_RELEASE(value);
  } else {
    OrtMemoryInfo* memoryInfo = nullptr;
    RETURN_NULLPTR_IF_ERROR(CreateCpuMemoryInfo, OrtDeviceAllocator, OrtMemTypeDefault, &memoryInfo);
    REGISTER_AUTO_RELEASE_HANDLE(MemoryInfo, memoryInfo);

    OrtValue* value = nullptr;
    int error_code = CHECK_STATUS(CreateTensorWithDataAsOrtValue, memoryInfo, data, data_length,
                                  dims_length > 0 ? shapes.data() : nullptr, dims_length,
                                  static_cast<ONNXTensorElementDataType>(data_type), &value);

    return (error_code == ORT_OK) ? value : nullptr;
  }
}

int OrtGetTensorData(OrtValue* tensor, int* data_type, void** data, size_t** dims, size_t* dims_length) {
  ONNXType tensor_type;
  RETURN_ERROR_CODE_IF_ERROR(GetValueType, tensor, &tensor_type);
  if (tensor_type != ONNX_TYPE_TENSOR) {
    return CheckStatus(
        Ort::GetApi().CreateStatus(ORT_NOT_IMPLEMENTED, "Reading data from non-tensor typed value is not supported."));
  }

  OrtTensorTypeAndShapeInfo* info = nullptr;
  RETURN_ERROR_CODE_IF_ERROR(GetTensorTypeAndShape, tensor, &info);
  REGISTER_AUTO_RELEASE_HANDLE(TensorTypeAndShapeInfo, info);

  size_t dims_len = 0;
  RETURN_ERROR_CODE_IF_ERROR(GetDimensionsCount, info, &dims_len);

  OrtAllocator* allocator = nullptr;
  RETURN_ERROR_CODE_IF_ERROR(GetAllocatorWithDefaultOptions, &allocator);

  size_t* p_dims = reinterpret_cast<size_t*>(allocator->Alloc(allocator, sizeof(size_t) * dims_len));
  REGISTER_AUTO_RELEASE_BUFFER(size_t, p_dims, allocator);

  ONNXTensorElementDataType type;
  RETURN_ERROR_CODE_IF_ERROR(GetTensorElementType, info, &type);

  std::vector<int64_t> shape(dims_len, 0);
  RETURN_ERROR_CODE_IF_ERROR(GetDimensions, info, shape.data(), shape.size());
  for (size_t i = 0; i < dims_len; i++) {
    p_dims[i] = static_cast<size_t>(shape[i]);
  }

  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    size_t num_elements;
    RETURN_ERROR_CODE_IF_ERROR(GetTensorShapeElementCount, info, &num_elements);

    // NOTE: ORT C-API does not expose an interface for users to get string raw data directly. There is always a copy.
    //       we can use the tensor raw data because it is type of "std::string *", which is very straightforward to
    //       implement and can also save memory usage. However, this approach depends on the Tensor's implementation
    //       details. So we have to copy the string content here.

    size_t string_data_length;
    RETURN_ERROR_CODE_IF_ERROR(GetStringTensorDataLength, tensor, &string_data_length);

    // The buffer contains following data:
    //  - a sequence of pointers to (const char*), size = num_elements * sizeof(const char*).
    //  - followed by a raw buffer to store string content, size = string_data_length + 1.
    size_t string_data_offset = num_elements * sizeof(const char*);
    size_t buf_size = string_data_offset + string_data_length;
    void* p_string_data = allocator->Alloc(allocator, buf_size + 1);
    void* p_string_content = reinterpret_cast<char*>(p_string_data) + string_data_offset;
    REGISTER_AUTO_RELEASE_BUFFER(void, p_string_data, allocator);

    size_t* p_offsets = reinterpret_cast<size_t*>(p_string_data);
    RETURN_ERROR_CODE_IF_ERROR(GetStringTensorContent, tensor, p_string_content, string_data_length, p_offsets, num_elements);

    // replace offsets by pointers
    const char** p_c_strs = reinterpret_cast<const char**>(p_offsets);
    for (size_t i = 0; i < num_elements; i++) {
      p_c_strs[i] = reinterpret_cast<const char*>(p_string_content) + p_offsets[i];
    }

    // put null at the last char
    reinterpret_cast<char*>(p_string_data)[buf_size] = '\0';

    *data = UNREGISTER_AUTO_RELEASE(p_string_data);
  } else {
    void* p_tensor_raw_data = nullptr;
    RETURN_ERROR_CODE_IF_ERROR(GetTensorMutableData, tensor, &p_tensor_raw_data);
    *data = p_tensor_raw_data;
  }

  *data_type = static_cast<int>(type);
  *dims_length = dims_len;
  *dims = UNREGISTER_AUTO_RELEASE(p_dims);
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
  REGISTER_AUTO_RELEASE_HANDLE(RunOptions, run_options);

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

  return UNREGISTER_AUTO_RELEASE(run_options);
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
#if defined(USE_JSEP)
  EM_ASM({ Module["jsepRunPromise"] = new Promise(function(r) { Module.jsepRunPromiseResolve = r; }); });
#endif
  auto status_code = CHECK_STATUS(Run, session, run_options, input_names, inputs, input_count, output_names, output_count, outputs);
#if defined(USE_JSEP)
  EM_ASM({ Module.jsepRunPromiseResolve($0); }, status_code);
#endif
  return status_code;
}

char* OrtEndProfiling(ort_session_handle_t session) {
  OrtAllocator* allocator = nullptr;
  RETURN_NULLPTR_IF_ERROR(GetAllocatorWithDefaultOptions, &allocator);

  char* file_name = nullptr;
  return (CHECK_STATUS(SessionEndProfiling, session, allocator, &file_name) == ORT_OK)
             ? file_name
             : nullptr;
}

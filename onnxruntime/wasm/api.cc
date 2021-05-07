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

int OrtInit(int level) {
  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
  switch (level) {
    case 0:
      level = ORT_LOGGING_LEVEL_VERBOSE;
      break;
    case 1:
      level = ORT_LOGGING_LEVEL_INFO;
      break;
    case 2:
      level = ORT_LOGGING_LEVEL_WARNING;
      break;
    case 3:
      level = ORT_LOGGING_LEVEL_ERROR;
      break;
    case 4:
      level = ORT_LOGGING_LEVEL_FATAL;
      break;
  }
  return CheckStatus(Ort::GetApi().CreateEnv(logging_level, "Default", &g_env));
}

OrtSessionOptions* OrtCreateSessionOptions() {
  OrtSessionOptions* session_options = nullptr;
  int error_code = CheckStatus(Ort::GetApi().CreateSessionOptions(&session_options));
  return (error_code == ORT_OK) ? session_options : nullptr;
}

void OrtReleaseSessionOptions(OrtSessionOptions* session_options) {
  Ort::OrtRelease(session_options);
}

int OrtSetSessionGraphOptimizationLevel(OrtSessionOptions* session_options, size_t level) {
  OrtStatusPtr status = nullptr;
  switch (level) {
    case 0:
      status = Ort::GetApi().SetSessionGraphOptimizationLevel(session_options, ORT_DISABLE_ALL);
      break;
    case 1:
      status = Ort::GetApi().SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);
      break;
    case 2:
      status = Ort::GetApi().SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);
      break;
    case 99:
    default:
      status = Ort::GetApi().SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);
      break;
  }
  return CheckStatus(status);
}

int OrtEnableCpuMemArena(OrtSessionOptions* session_options) {
  return CheckStatus(Ort::GetApi().EnableCpuMemArena(session_options));
}

int OrtDisableCpuMemArena(OrtSessionOptions* session_options) {
  return CheckStatus(Ort::GetApi().DisableCpuMemArena(session_options));
}

int OrtEnableMemPattern(OrtSessionOptions* session_options) {
  return CheckStatus(Ort::GetApi().EnableMemPattern(session_options));
}

int OrtDisableMemPattern(OrtSessionOptions* session_options) {
  return CheckStatus(Ort::GetApi().DisableMemPattern(session_options));
}

int OrtSetSessionExecutionMode(OrtSessionOptions* session_options, size_t mode) {
  OrtStatusPtr status = nullptr;
  switch (mode) {
    case 1:
      status = Ort::GetApi().SetSessionExecutionMode(session_options, ORT_PARALLEL);
      break;
    case 0:
    default:
      status = Ort::GetApi().SetSessionExecutionMode(session_options, ORT_SEQUENTIAL);
      break;
  }
  return CheckStatus(status);
}

int OrtSetSessionLogId(OrtSessionOptions* session_options, const char* logid) {
  return CheckStatus(Ort::GetApi().SetSessionLogId(session_options, logid));
}

int OrtSetSessionLogSeverityLevel(OrtSessionOptions* session_options, size_t level) {
  return CheckStatus(Ort::GetApi().SetSessionLogSeverityLevel(session_options, level));
}

OrtSession* OrtCreateSession(void* data, size_t data_length, OrtSessionOptions* session_options) {
  OrtSession* session = nullptr;
  int error_code = ORT_OK;

#if !defined(__EMSCRIPTEN_PTHREADS__)
  if (session_options) {
    // must disable thread pool when WebAssembly multi-threads support is disabled.
    error_code = CheckStatus(Ort::GetApi().SetIntraOpNumThreads(session_options, 1));
    if (error_code != ORT_OK) {
      return nullptr;
    }
    error_code = CheckStatus(Ort::GetApi().SetSessionExecutionMode(session_options, ORT_SEQUENTIAL));
    if (error_code != ORT_OK) {
      return nullptr;
    }
  }
#endif

  error_code = CheckStatus(Ort::GetApi().CreateSessionFromArray(g_env, data, data_length, session_options, &session));
  return (error_code == ORT_OK) ? session : nullptr;
}

void OrtReleaseSession(OrtSession* session) {
  Ort::OrtRelease(session);
}

size_t OrtGetInputCount(OrtSession* session) {
  size_t input_count = 0;
  int error_code = CheckStatus(Ort::GetApi().SessionGetInputCount(session, &input_count));
  return (error_code == ORT_OK) ? input_count : 0;
}

size_t OrtGetOutputCount(OrtSession* session) {
  size_t output_count = 0;
  int error_code = CheckStatus(Ort::GetApi().SessionGetOutputCount(session, &output_count));
  return (error_code == ORT_OK) ? output_count : 0;
}

char* OrtGetInputName(OrtSession* session, size_t index) {
  OrtAllocator* allocator = nullptr;
  int error_code = CheckStatus(Ort::GetApi().GetAllocatorWithDefaultOptions(&allocator));
  if (error_code != ORT_OK) {
    return nullptr;
  }
  char* input_name = nullptr;
  error_code = CheckStatus(Ort::GetApi().SessionGetInputName(session, index, allocator, &input_name));
  return (error_code == ORT_OK) ? input_name : nullptr;
}

char* OrtGetOutputName(OrtSession* session, size_t index) {
  OrtAllocator* allocator = nullptr;
  int error_code = CheckStatus(Ort::GetApi().GetAllocatorWithDefaultOptions(&allocator));
  if (error_code != ORT_OK) {
    return nullptr;
  }
  char* output_name = nullptr;
  error_code = CheckStatus(Ort::GetApi().SessionGetOutputName(session, index, allocator, &output_name));
  return (error_code == ORT_OK) ? output_name : nullptr;
}

void OrtFree(void* ptr) {
  Ort::AllocatorWithDefaultOptions allocator;
  allocator.Free(ptr);
}

OrtValue* OrtCreateTensor(int data_type, void* data, size_t data_length, size_t* dims, size_t dims_length) {
  std::vector<int64_t> shapes(dims_length);
  for (size_t i = 0; i < dims_length; i++) {
    shapes[i] = dims[i];
  }

  return Ort::Value::CreateTensor(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
                                  data,
                                  data_length,
                                  dims_length > 0 ? shapes.data() : nullptr,
                                  dims_length,
                                  static_cast<ONNXTensorElementDataType>(data_type))
      .release();
}

void OrtGetTensorData(OrtValue* tensor, int* data_type, void** data, size_t** dims, size_t* dims_length) {
  Ort::Value v{tensor};
  auto info = v.GetTensorTypeAndShapeInfo();
  size_t dims_len = info.GetDimensionsCount();
  Ort::AllocatorWithDefaultOptions allocator;
  size_t* p_dims = reinterpret_cast<size_t*>(allocator.Alloc(sizeof(size_t) * dims_len));
  *data = v.GetTensorMutableData<void>();
  *data_type = info.GetElementType();
  *dims_length = dims_len;
  auto shape = info.GetShape();
  for (size_t i = 0; i < dims_len; i++) {
    p_dims[i] = static_cast<size_t>(shape[i]);
  }
  *dims = p_dims;
  v.release();
}

void OrtReleaseTensor(OrtValue* tensor) {
  Ort::OrtRelease(tensor);
}

OrtRunOptions* OrtCreateRunOptions() {
  OrtRunOptions* run_options = nullptr;
  int error_code = CheckStatus(Ort::GetApi().CreateRunOptions(&run_options));
  return (error_code == ORT_OK) ? run_options : nullptr;
}

void OrtReleaseRunOptions(OrtRunOptions* run_options) {
  Ort::OrtRelease(run_options);
}

int OrtRunOptionsSetRunLogSeverityLevel(OrtRunOptions* run_options, size_t level) {
  return CheckStatus(Ort::GetApi().RunOptionsSetRunLogSeverityLevel(run_options, level));
}

int OrtRunOptionsSetRunTag(OrtRunOptions* run_options, const char* tag) {
  return CheckStatus(Ort::GetApi().RunOptionsSetRunTag(run_options, tag));
}

int OrtRun(OrtSession* session,
           const char** input_names, const ort_tensor_handle_t* inputs, size_t input_count,
           const char** output_names, size_t output_count, ort_tensor_handle_t* outputs,
           OrtRunOptions* run_options) {
  return CheckStatus(Ort::GetApi().Run(session, run_options, input_names, inputs, input_count, output_names, output_count, outputs));
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "api.h"

#include "core/session/onnxruntime_cxx_api.h"

#include <iostream>
#include <vector>

namespace {
Ort::Env* g_env;
}  // namespace

void OrtInit() {
  // TODO: allow user to specify logging
  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
  g_env = new Ort::Env{logging_level, "Default"};
}

Ort::Session* OrtCreateSession(void* data, size_t data_length) {
  Ort::SessionOptions session_options;
  session_options.SetLogId("onnxruntime");

  // disable thread pool for now since not all major browsers support WebAssembly threading.
  session_options.SetIntraOpNumThreads(1);

  return new Ort::Session(*g_env, data, data_length, session_options);
}

void OrtReleaseSession(Ort::Session* session) {
  delete session;
}

size_t OrtGetInputCount(Ort::Session* session) {
  return session->GetInputCount();
}

size_t OrtGetOutputCount(Ort::Session* session) {
  return session->GetOutputCount();
}

char* OrtGetInputName(Ort::Session* session, size_t index) {
  Ort::AllocatorWithDefaultOptions allocator;
  return session->GetInputName(index, allocator);
}

char* OrtGetOutputName(Ort::Session* session, size_t index) {
  Ort::AllocatorWithDefaultOptions allocator;
  return session->GetOutputName(index, allocator);
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

  return Ort::Value::CreateTensor({},
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

int OrtRun(Ort::Session* session,
           const char** input_names, const ort_tensor_handle_t* inputs, size_t input_count,
           const char** output_names, size_t output_count, ort_tensor_handle_t* outputs) {
  OrtStatusPtr status = Ort::GetApi().Run(*session, Ort::RunOptions{nullptr}, input_names, inputs, input_count, output_names, output_count, outputs);
  OrtErrorCode error_code = ORT_OK;
  if (status) {
    std::string error_message = Ort::GetApi().GetErrorMessage(status);
    error_code = Ort::GetApi().GetErrorCode(status);
    std::cerr << Ort::Exception(std::move(error_message), error_code).what()
              << std::endl;
    Ort::GetApi().ReleaseStatus(status);
  }
  return error_code;
}

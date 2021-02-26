#include "api.h"

#include "core/session/onnxruntime_cxx_api.h"

#include <vector>

namespace {
Ort::Env* g_env;
}  // namespace

void ort_init() {
  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
  g_env = new Ort::Env{logging_level, "Default"};
}

Ort::Session* ort_create_session(void* data, size_t data_length) {
  Ort::SessionOptions session_options;
  session_options.SetLogId("onnxjs");
  session_options.SetIntraOpNumThreads(1);

  return new Ort::Session(*g_env, data, data_length, session_options);
}

void ort_release_session(Ort::Session* session) {
  Ort::Session* p = session;
  delete p;
}

size_t ort_get_input_count(Ort::Session* session) {
  return session->GetInputCount();
}
size_t ort_get_output_count(Ort::Session* session) {
  return session->GetOutputCount();
}

char* ort_get_input_name(Ort::Session* session, size_t index) {
  Ort::AllocatorWithDefaultOptions a;
  return session->GetInputName(index, a);
}

char* ort_get_output_name(Ort::Session* session, size_t index) {
  Ort::AllocatorWithDefaultOptions a;
  return session->GetOutputName(index, a);
}

void ort_free(void* ptr) {
  Ort::AllocatorWithDefaultOptions a;
  a.Free(ptr);
}

OrtValue* ort_create_tensor(int data_type, void* data, size_t data_length, size_t* dims, size_t dims_length) {
  std::vector<int64_t> shapes(dims_length);
  for (size_t i = 0; i < dims_length; i++) {
    shapes[i] = dims[i];
  }

  return Ort::Value::CreateTensor({}, data, data_length, dims_length > 0 ? shapes.data() : nullptr, dims_length, static_cast<ONNXTensorElementDataType>(data_type))
      .release();
}

void ort_get_tensor_data(OrtValue* tensor, int* data_type, void** data, size_t** dims, size_t* dims_length) {
  Ort::Value v{tensor};
  auto info = v.GetTensorTypeAndShapeInfo();
  size_t dims_len = info.GetDimensionsCount();
  Ort::AllocatorWithDefaultOptions a;
  size_t* p_dims = reinterpret_cast<size_t*>(a.Alloc(sizeof(size_t) * dims_len));
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

void ort_release_tensor(OrtValue* tensor) {
  Ort::OrtRelease(tensor);
}

void ort_run(ort_session_handle_t session, const char** input_names, const ort_tensor_handle_t* inputs, size_t input_count, const char** output_names, size_t output_count, ort_tensor_handle_t* outputs) {
  Ort::ThrowOnError(Ort::GetApi().Run(*session, Ort::RunOptions{nullptr}, input_names, inputs, input_count, output_names, output_count, outputs));
}

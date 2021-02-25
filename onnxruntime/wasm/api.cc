#include "api.h"

#include "core/session/onnxruntime_cxx_api.h"

#include <vector>

namespace {
static Ort::Env* g_env;

struct TensorMetadata {
  ONNXTensorElementDataType data_type;
  size_t data_size;
  void* data;
  size_t shape_len;

  size_t* shapes() const { return reinterpret_cast<size_t*>(reinterpret_cast<size_t>(this) + sizeof(TensorMetadata)); }
};
static_assert(sizeof(TensorMetadata) == sizeof(size_t) + sizeof(size_t) + sizeof(void*) + sizeof(size_t), "memory is not aligned");

struct Feed {
  const char* name;
  ort_tensor_t tensor;
};

struct RunContext {
  size_t input_count;
  size_t output_count;

  Feed* feeds() const { return reinterpret_cast<Feed*>(reinterpret_cast<size_t>(this) + sizeof(RunContext)); }
};
static_assert(sizeof(RunContext) == sizeof(size_t) + sizeof(size_t), "memory is not aligned");

}  // namespace

void ort_init() {
  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
  g_env = new Ort::Env{logging_level, "Default"};
}

ort_session_handle_t ort_create_session(ort_model_data_t data) {
  size_t* p_size = reinterpret_cast<size_t*>(data);
  void* p_data = reinterpret_cast<void*>(data + sizeof(size_t));

  Ort::SessionOptions session_options;
  session_options.SetLogId("onnxjs");
  session_options.SetIntraOpNumThreads(1);

  return reinterpret_cast<ort_session_handle_t>(new Ort::Session(*g_env, p_data, *p_size, session_options));
}

void ort_release_session(ort_session_handle_t session) {
  Ort::Session* p = reinterpret_cast<Ort::Session*>(session);
  delete p;
}

ort_tensor_t ort_create_tensor(ort_tensor_metadata_t metadata) {
  TensorMetadata* p_metadata = reinterpret_cast<TensorMetadata*>(metadata);
  size_t shape_len = p_metadata->shape_len;
  size_t* p_shapes = p_metadata->shapes();
  std::vector<int64_t> dims(shape_len);
  for (size_t i = 0; i < shape_len; i++) {
    dims[i] = p_shapes[i];
  }

  return reinterpret_cast<ort_tensor_t>(
      Ort::Value::CreateTensor({}, p_metadata->data, p_metadata->data_size, shape_len > 0 ? dims.data() : nullptr, shape_len, p_metadata->data_type)
          .release());
}

ort_tensor_metadata_t ort_get_tensor_metadata(ort_tensor_t tensor) {
  Ort::Value v{reinterpret_cast<OrtValue*>(tensor)};
  auto info = v.GetTensorTypeAndShapeInfo();
  size_t dims_len = info.GetDimensionsCount();
  size_t metadata_size = sizeof(TensorMetadata) + sizeof(size_t) * dims_len;
  TensorMetadata* p = reinterpret_cast<TensorMetadata*>(malloc(metadata_size));
  p->data = v.GetTensorMutableData<void>();
  p->data_type = info.GetElementType();
  p->shape_len = dims_len;
  auto shape = info.GetShape();
  for (size_t i = 0; i < dims_len; i++) {
    p->shapes()[i] = static_cast<size_t>(shape[i]);
  }

  v.release();
  return reinterpret_cast<ort_tensor_metadata_t>(p);
}

void ort_release_tensor_metadata(ort_tensor_metadata_t metadata) {
  free(reinterpret_cast<void*>(metadata));
}

void ort_release_tensor(ort_tensor_t tensor) {
  OrtValue* value = reinterpret_cast<OrtValue*>(tensor);
  Ort::GetApi().ReleaseValue(value);
}

void ort_run(ort_session_handle_t p_session, ort_run_context_t context) {
  RunContext* p_context = reinterpret_cast<RunContext*>(context);
  size_t input_count = p_context->input_count;
  size_t output_count = p_context->output_count;
  Feed* feeds = p_context->feeds();

  std::vector<Ort::Value> inputs;
  inputs.reserve(input_count);
  std::vector<const char*> input_names(input_count);
  std::vector<const char*> output_names(output_count);

  for (size_t i = 0; i < input_count; i++) {
    const char* n = feeds[i].name;
    ort_tensor_t t = feeds[i].tensor;
    inputs.emplace_back(Ort::Value{reinterpret_cast<OrtValue*>(t)});
    input_names[i] = n;
  }
  for (size_t i = 0; i < output_count; i++) {
    const char* n = feeds[input_count + i].name;
    output_names[i] = n;
  }

  auto session = reinterpret_cast<Ort::Session*>(p_session);
  Ort::AllocatorWithDefaultOptions allocator;

  auto outputs = session->Run(Ort::RunOptions{nullptr},
                              input_count ? input_names.data() : nullptr,
                              input_count ? inputs.data() : nullptr,
                              input_count,
                              output_count ? output_names.data() : nullptr,
                              output_count);

  for (size_t i = 0; i < input_count; i++) {
    inputs[i].release();
  }
  for (size_t i = 0; i < output_count; i++) {
    feeds[input_count + i].tensor = reinterpret_cast<ort_tensor_t>(outputs[i].release());
  }
}

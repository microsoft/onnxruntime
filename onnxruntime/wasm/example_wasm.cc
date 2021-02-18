#include <iostream>

#include "core/session/onnxruntime_cxx_api.h"
#include "example_wasm.h"

const OrtApi* ort_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);

bool Example::Load(const emscripten::val& model_jsarray) {
  try {
    OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
    env_.reset(new Ort::Env{logging_level, "Default"});
  } catch (const Ort::Exception& e) {
    std::cerr << "Can't create environment: " << e.what() << std::endl;
    return false;
  };

  std::vector<uint8_t> model_data = emscripten::vecFromJSArray<uint8_t>(model_jsarray);

  try {
    Ort::SessionOptions session_options;
    session_options.SetLogId("onnxjs");
    session_options.SetIntraOpNumThreads(1);

    session_.reset(new Ort::Session(*env_,
                                    static_cast<void*>(model_data.data()),
                                    static_cast<int>(model_data.size()),
                                    session_options));
  } catch (const Ort::Exception& e) {
    std::cerr << "Can't create session: " << e.what() << std::endl;
    return false;
  }

  return true;
}

bool Example::Run() {
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<const char*> input_names{"A", "B"};
  const char* output_names[] = {"C"};

  std::vector<Ort::Value> values;
  for (size_t i = 0; i < input_names.size(); ++i) {
    float data[] = {1., 2., 3., 4., 5.};
    const int data_len = sizeof(data) / sizeof(data[0]);
    const int64_t shape[] = {5};
    const size_t shape_len = sizeof(shape) / sizeof(shape[0]);
    values.emplace_back(Ort::Value::CreateTensor<float>(mem_info, data, data_len, shape, shape_len));
  }

  auto outputs = session_->Run(Ort::RunOptions{nullptr},
                               input_names.data(),
                               values.data(),
                               input_names.size(),
                               output_names,
                               1);

  const auto& output = outputs[0];
  auto type_info = output.GetTensorTypeAndShapeInfo();
  for (int i = 0; i < type_info.GetElementCount(); ++i) {
    std::cout << output.GetTensorData<float>()[i] << std::endl;
  }

  return true;
}

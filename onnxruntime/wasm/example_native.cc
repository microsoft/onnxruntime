#include <fstream>
#include <iostream>

#include "core/session/onnxruntime_cxx_api.h"
#include "example_native.h"

bool Example::Load(const std::string& model_path) {
  try {
    OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
    env_.reset(new Ort::Env{logging_level, "Default"});
  } catch (const Ort::Exception& e) {
    std::cerr << "Can't create environment: " << e.what() << std::endl;
    return false;
  };

  std::streampos file_size;
  std::ifstream ifs(model_path, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Can't open a model: " << model_path << std::endl;
    return false;
  }
  ifs.seekg(0, std::ios::end);
  file_size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<char> model_data(file_size);
  ifs.read((char*)&model_data[0], file_size);

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

std::vector<Ort::Value> Example::Run(const char* const* input_names,
                                     const Ort::Value* input_values,
                                     size_t input_count,
                                     const char* const* output_names,
                                     size_t output_count) {
  return session_->Run(Ort::RunOptions{nullptr},
                       input_names,
                       input_values,
                       input_count,
                       output_names,
                       output_count);
}

int main(int argc, char** argv) {
  Example example;
  example.Load(argv[1]);

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<const char*> input_names{"A", "B"};
  const char* output_names[] = {"C"};
  std::vector<float> data{1., 2., 3., 4., 5.};
  std::vector<int64_t> shape{5};

  std::vector<Ort::Value> values;
  for (size_t i = 0; i < input_names.size(); ++i) {
    values.emplace_back(Ort::Value::CreateTensor<float>(mem_info,
                                                        data.data(),
                                                        data.size(),
                                                        shape.data(),
                                                        shape.size()));
  }

  auto outputs = example.Run(input_names.data(), values.data(), input_names.size(), output_names, 1);

  const auto& output = outputs[0];
  auto type_info = output.GetTensorTypeAndShapeInfo();
  for (int i = 0; i < type_info.GetElementCount(); ++i) {
    std::cout << output.GetTensorData<float>()[i] << std::endl;
  }

  return 0;
}

#if defined(BUILD_NATIVE)
#include <fstream>
#endif
#include <iostream>
#include <numeric>

#include "core/common/common.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "example.h"

#if defined(BUILD_NATIVE)
bool Example::Load(const std::string& model_path) {
#else
bool Example::Load(const emscripten::val& model_jsarray) {
#endif
  ORT_TRY {
    OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
    env_.reset(new Ort::Env{logging_level, "Default"});
  }
  ORT_CATCH(const Ort::Exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << "Can't create environment: " << e.what() << std::endl;
    });
  };

#if defined(BUILD_NATIVE)
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
#else
  std::vector<uint8_t> model_data = emscripten::vecFromJSArray<uint8_t>(model_jsarray);
#endif

  ORT_TRY {
    Ort::SessionOptions session_options;
    session_options.SetLogId("onnxjs");
    session_options.SetIntraOpNumThreads(1);

    session_.reset(new Ort::Session(*env_,
                                    static_cast<void*>(model_data.data()),
                                    static_cast<int>(model_data.size()),
                                    session_options));
  }
  ORT_CATCH(const Ort::Exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << "Can't create session: " << e.what() << std::endl;
    });
  }

  return true;
}

bool Example::Run() {
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<const char*> input_names{"bgrm"};
  const char* output_names[] = {"mask"};
  std::vector<int64_t> shape{1, 4, 160, 160};
  size_t data_size = std::accumulate(shape.begin(), shape.end(), 1, [](int size, int n) {
    return size * n;
  });
  std::vector<float> data(data_size, 1.0);

  std::vector<Ort::Value> values;
  for (size_t i = 0; i < input_names.size(); ++i) {
    values.emplace_back(Ort::Value::CreateTensor<float>(mem_info,
                                                        data.data(),
                                                        data_size,
                                                        shape.data(),
                                                        shape.size()));
  }

  auto outputs = session_->Run(Ort::RunOptions{nullptr},
                               input_names.data(),
                               values.data(),
                               input_names.size(),
                               output_names,
                               1);

  const auto& output = outputs[0];
  auto type_info = output.GetTensorTypeAndShapeInfo();
  std::cout << "Output count: " << type_info.GetElementCount() << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << output.GetTensorData<float>()[i] << std::endl;
  }

  return true;
}

#if defined(BUILD_NATIVE)
int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " model" << std::endl;
    return -1;
  }

  Example example;
  example.Load(argv[1]);
  example.Run();
  return 0;
}
#endif

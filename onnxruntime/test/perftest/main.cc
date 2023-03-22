// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/session/onnxruntime_c_api.h>
#include <random>
#include "command_args_parser.h"
#include "performance_runner.h"
#include <google/protobuf/stubs/common.h>

using namespace onnxruntime;
const OrtApi* g_ort = NULL;

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[]) {
#else
int real_main(int argc, char* argv[]) {
#endif
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  perftest::PerformanceTestConfig test_config;
  if (!perftest::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    perftest::CommandLineParser::ShowUsage();
    return -1;
  }
  std::ifstream fs("c_sample.txt");
  if (fs.is_open()) {
    Ort::Env env;

    int ep_global_index = -1;
    std::unordered_map<std::string, std::string> provider_options;
    env.CreateAndRegisterExecutionProvider(true, "CPUExecutionProvider", provider_options, &ep_global_index);
    Ort::Session session(env, L"C:/share/models/Detection/model.onnx", &ep_global_index, 1);
    Ort::AllocatorWithDefaultOptions allocator;
    auto input0 = session.GetInputNameAllocated(0, allocator);
    std::vector<const char*> inputNames{input0.get()};
    std::vector<int64_t> shape0{1, 3, 256, 256};
    int data_length = 3 * 256 * 256;
    float* inputData = new float[data_length];
    for (int i = 0; i < data_length; i++) inputData[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    Ort::Value tensor = Ort::Value::CreateTensor(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault), inputData, data_length * 4, shape0.data(), shape0.size());
    Ort::AllocatorWithDefaultOptions allocatorOut;
    auto output0 = session.GetOutputNameAllocated(0, allocatorOut);
    std::vector<const char*> outputNames{output0.get()};

    auto ret = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &tensor, inputNames.size(), outputNames.data(), 1);

#ifdef USE_XNNPACK
    int xnnpack_global_index = -1;
    env.CreateAndRegisterExecutionProvider(true, "XnnpackExecutionProvider", provider_options, &xnnpack_global_index);
    int global_ep_index[2] = {xnnpack_global_index, ep_global_index};
    Ort::Session session2(env, L"C:/share/models/Detection/model.onnx", global_ep_index, 2);
    auto ret2 = session2.Run(Ort::RunOptions{nullptr}, inputNames.data(), &tensor, inputNames.size(), outputNames.data(), 1);
#endif  // USE_XNNPACK
    
    fs.close();
  } else {
    Ort::Env env{nullptr};
    {
      bool failed = false;
      ORT_TRY {
        OrtLoggingLevel logging_level = test_config.run_config.f_verbose
                                            ? ORT_LOGGING_LEVEL_VERBOSE
                                            : ORT_LOGGING_LEVEL_WARNING;
        env = Ort::Env(logging_level, "Default");
      }
      ORT_CATCH(const Ort::Exception& e) {
        ORT_HANDLE_EXCEPTION([&]() {
          fprintf(stderr, "Error creating environment: %s \n", e.what());
          failed = true;
        });
      }

      if (failed)
        return -1;
    }
    std::random_device rd;
    perftest::PerformanceRunner perf_runner(env, test_config, rd);
    auto status = perf_runner.Run();
    if (!status.IsOK()) {
      printf("Run failed:%s\n", status.ErrorMessage().c_str());
      return -1;
    }

    perf_runner.SerializeResult();
  }
  return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  int retval = -1;
  ORT_TRY {
    retval = real_main(argc, argv);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      fprintf(stderr, "%s\n", ex.what());
      retval = -1;
    });
  }

  ::google::protobuf::ShutdownProtobufLibrary();

  return retval;
}

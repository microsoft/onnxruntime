// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/session/onnxruntime_c_api.h>
#include <random>
#include "command_args_parser.h"
#include "performance_runner.h"
#include "utils.h"
#include "strings_helper.h"
#include <google/protobuf/stubs/common.h>

using namespace onnxruntime;
const OrtApi* g_ort = NULL;

int RunPerfTest(Ort::Env& env, const perftest::PerformanceTestConfig& test_config);
Ort::Status CompileEpContextModel(const Ort::Env& env, const perftest::PerformanceTestConfig& test_config);

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[]) {
#else
int real_main(int argc, char* argv[]) {
#endif
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  perftest::PerformanceTestConfig test_config;
  if (!perftest::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    fprintf(stderr, "%s", "See 'onnxruntime_perf_test --help'.");
    return -1;
  }
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
        std::cerr << "Error creating environment: " << e.what() << std::endl;
        failed = true;
      });
    }

    if (failed)
      return -1;
  }

  if (!test_config.plugin_ep_names_and_libs.empty()) {
    perftest::utils::RegisterExecutionProviderLibrary(env, test_config);
  }

  // Unregister all registered plugin EP libraries before program exits.
  // This is necessary because unregistering the plugin EP also unregisters any associated shared allocators.
  // If we don't do this and program returns, the factories stored inside the environment will be destroyed when the environment goes out of scope.
  // Later, when the shared allocator's deleter runs, it may cause a segmentation fault because it attempts to use the already-destroyed factory to call ReleaseAllocator.
  // See "ep_device.ep_factory->ReleaseAllocator" in Environment::CreateSharedAllocatorImpl.
  auto unregister_plugin_eps_at_scope_exit = gsl::finally([&]() {
    if (!test_config.registered_plugin_eps.empty()) {
      perftest::utils::UnregisterExecutionProviderLibrary(env, test_config);  // this won't throw
    }
  });

  if (test_config.list_available_ep_devices) {
    perftest::utils::ListEpDevices(env);
    if (test_config.registered_plugin_eps.empty()) {
      fprintf(stdout, "No plugin execution provider libraries are registered. Please specify them using \"--plugin_ep_libs\"; otherwise, only CPU may be available.\n");
    }
    return 0;
  }

  int status = 0;

  // EP context perf test
  if (test_config.run_config.compile_ep_context) {
    {
      std::cout << "\n> Compiling model...\n";
      auto compile_status = CompileEpContextModel(env, test_config);

      if (!compile_status.IsOK())
        return -1;
    }

    {
      test_config.model_info.model_file_path = test_config.run_config.compile_model_path;
      status = RunPerfTest(env, test_config);
    }
  } else {
    // regular perf test
    status = RunPerfTest(env, test_config);
  }
  return status;
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
      std::cerr << ex.what() << std::endl;
      retval = -1;
    });
  }

  ::google::protobuf::ShutdownProtobufLibrary();

  return retval;
}

int RunPerfTest(Ort::Env& env, const perftest::PerformanceTestConfig& test_config) {
  std::random_device rd;
  perftest::PerformanceRunner perf_runner(env, test_config, rd);

  // Exit if user enabled -n option so that user can measure session creation time
  if (test_config.run_config.exit_after_session_creation) {
    perf_runner.LogSessionCreationTime();
    return 0;
  }

  auto status = perf_runner.Run();
  if (!status.IsOK()) {
    printf("Run failed:%s\n", status.ErrorMessage().c_str());
    return -1;
  }

  perf_runner.SerializeResult();
  return 0;
}

struct CustomInitializerHandlerState {
  const ORTCHAR_T* external_file_path = nullptr;
  std::ofstream* outfile = nullptr;
};

static OrtStatus* ORT_API_CALL TestHandleInitializerDataFunc(void* state,
                                                             const char* initializer_name,
                                                             const OrtValue* c_initializer_value,
                                                             const OrtExternalInitializerInfo* /*c_external_info*/,
                                                             OrtExternalInitializerInfo** c_new_external_info) {
  Ort::Status final_status{nullptr};

  ORT_TRY {
    CustomInitializerHandlerState* custom_state = reinterpret_cast<CustomInitializerHandlerState*>(state);

    //
    // Store other initializers in an external file.
    //
    Ort::ConstValue value{c_initializer_value};
    size_t byte_size = value.GetTensorSizeInBytes();

    int64_t offset = custom_state->outfile->tellp();
    const ORTCHAR_T* location = custom_state->external_file_path;

    custom_state->outfile->write(static_cast<const char*>(value.GetTensorRawData()), byte_size);
    custom_state->outfile->flush();
    

    // Provide caller (ORT) with the new external info.
    Ort::ExternalInitializerInfo new_external_info{nullptr};
    if (Ort::Status status = Ort::ExternalInitializerInfo::Create(location, offset, byte_size, new_external_info);
        !status.IsOK()) {
      return status.release();
    }

    *c_new_external_info = new_external_info.release();
  }
  ORT_CATCH(const Ort::Exception& ex) {
    ORT_HANDLE_EXCEPTION(([&ex, &final_status]() {
      final_status = Ort::Status{ex};
    }));
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION(([&ex, &final_status]() {
      final_status = Ort::Status(ex.what(), ORT_FAIL);
    }));
  }

  return final_status.release();
}

Ort::Status CompileEpContextModel(const Ort::Env& env, const perftest::PerformanceTestConfig& test_config) {
  auto output_ctx_model_path = test_config.run_config.compile_model_path;
  const auto provider_name = test_config.machine_config.provider_type_name;

  Ort::SessionOptions session_options;

  std::unordered_map<std::string, std::string> provider_options;
  session_options.AppendExecutionProvider(provider_name, provider_options);

  // Open a file to store external initializers. ORT will call our handler function for every initializer.
  // const ORTCHAR_T* initializer_file = ORT_TSTR("./nv_execution_provider_compile_large_embed_mode_0_bytestream_io_0_ext_init_0.onnx_data");
  // CustomInitializerHandlerState custom_state = {initializer_file};
  const ORTCHAR_T* initializer_file = ORT_TSTR("./nv_execution_provider_external_weights.onnx_data");
  std::ofstream outfile(initializer_file, std::ios::binary);
  CustomInitializerHandlerState custom_state = {initializer_file, &outfile};

  Ort::ModelCompilationOptions model_compile_options(env, session_options);
  model_compile_options.SetEpContextEmbedMode(test_config.run_config.compile_binary_embed);
  model_compile_options.SetInputModelPath(test_config.model_info.model_file_path.c_str());
  model_compile_options.SetOutputModelPath(output_ctx_model_path.c_str());
  model_compile_options.SetOutputModelGetInitializerLocationFunc(TestHandleInitializerDataFunc,
                                                                 reinterpret_cast<void*>(&custom_state));

  Ort::Status status;
  std::chrono::duration<double> compile_duration;
  {
    auto compile_time_start = std::chrono::high_resolution_clock::now();
    status = Ort::CompileModel(env, model_compile_options);
    auto compile_time_end = std::chrono::high_resolution_clock::now();
    compile_duration = compile_time_end - compile_time_start;
  }

  if (!status.IsOK()) {
    std::cout << "Failed to compile model: " << status.GetErrorMessage() << std::endl;
  } else {
    std::cout << "Compile time cost: " << compile_duration.count() << " s\n";
  }
  return status;
}

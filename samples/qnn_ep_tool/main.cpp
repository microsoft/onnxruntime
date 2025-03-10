// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnxruntime_cxx_api.h>

#include <filesystem> // NOLINT
#include <iostream>
#include <string>
#include <vector>

#include "core/platform/path_lib.h"
#include "include/utils.hpp"

int main(int, char* argv[]) {
  std::string model_dir(argv[1]);
  std::cout << model_dir << std::endl;
  std::basic_string<PATH_CHAR_TYPE> model_path = find_model_path(model_dir);
  if (model_path.size() <= 0) {
    std::cout << ".onnx model should be provided" << std::endl;
    exit(0);
  }

  // model
  std::cout << "[Successfully Load Model] " << std::endl;
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  std::cout << "[ORT_API_VERSION] " << ORT_API_VERSION << std::endl;
  OrtEnv* env;
  g_ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "test", &env);
  OrtSessionOptions* session_options;
  g_ort->CreateSessionOptions(&session_options);
  g_ort->SetIntraOpNumThreads(session_options, 1);
  g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

  std::string backend_path(argv[2]);
  if (!std::filesystem::exists(std::filesystem::path(backend_path))) {
    std::cout << "Not Found:" << backend_path << std::endl;
    exit(0);
  }
  std::vector<const char*> options_keys = {"backend_path"};
  std::vector<const char*> options_values = {backend_path.c_str()};

  g_ort->SessionOptionsAppendExecutionProvider(
    session_options, "QNN",
    options_keys.data(), options_values.data(), options_keys.size()
  );

  OrtSession* session;
  g_ort->CreateSession(env, model_path.c_str(), session_options, &session);
  std::cout << "[Successfully CreateSession]" << std::endl;

  OrtAllocator* allocator;
  g_ort->GetAllocatorWithDefaultOptions(&allocator);

  OnnxModelInfo model_info(g_ort, session, allocator);
  model_info.PrintOnnxModelInfo();

  // Multiple test_data_set_X
  std::vector<std::basic_string<PATH_CHAR_TYPE>> test_data_sets = find_test_data_sets(model_dir);
  std::cout << "test_data_sets.size() " << test_data_sets.size() << std::endl;
  for (size_t idx = 0; idx < test_data_sets.size(); idx++) {
    std::cout << "---- test_data_set_" << idx << " ----" << std::endl;
    std::vector<std::vector<float>> input_data;
    auto test_data_set_dir = std::filesystem::path(test_data_sets[idx]);
    auto data_format = check_data_format(test_data_set_dir);
    if (data_format == "pb") {
      std::cout << "[test_data_sets_" << idx << "] " << "Loading .pb" << std::endl;
      load_input_tensors_from_pbs(
        test_data_set_dir,
        g_ort,
        &model_info,
        &input_data
      );
    } else if (data_format == "raw") {
      std::cout << "[test_data_sets_" << idx << "] " << "Loading .raw" << std::endl;
      load_input_tensors_from_raws(
        test_data_set_dir,
        g_ort,
        &model_info,
        &input_data
      );
    }
    std::cout << "[test_data_sets_" << idx << "] " << "Successfully Load Inputs" << std::endl;
    g_ort->Run(
      session,
      nullptr,
      model_info.get_in_tensor_names().data(),
      (const OrtValue* const*)model_info.get_in_tensors().data(),
      model_info.get_in_tensors().size(),
      model_info.get_out_tensor_names().data(),
      model_info.get_out_tensor_names().size(),
      model_info.get_out_tensors().data()
    );
    std::cout << "[test_data_sets_" << idx << "] " << "Successfully Inference" << std::endl;
    if (data_format == "pb") {
      std::cout << "[test_data_sets_" << idx << "] " << "Dumping .pb" << std::endl;
      dump_output_tensors_to_pbs(
        test_data_set_dir,
        g_ort,
        &model_info
      );
    } else if (data_format == "raw") {
      std::cout << "[test_data_sets_" << idx << "] " << "Dumping .raw" << std::endl;
      dump_output_tensors_to_raws(
        test_data_set_dir,
        g_ort,
        &model_info
      );
    }
    std::cout << "[test_data_sets_" << idx << "] " << "Successfully Save Outputs" << std::endl;
    model_info.release_ort_values(g_ort);
    std::cout << "[test_data_sets_" << idx << "] " << "Successfully Release OrtValue" << std::endl;
  }
  return 0;
}

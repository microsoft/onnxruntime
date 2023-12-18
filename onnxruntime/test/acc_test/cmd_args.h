// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <string_view>
#include <filesystem>

struct AppArgs {
  std::filesystem::path test_dir;
  std::string output_file;
  std::filesystem::path expected_accuracy_file;
  std::string execution_provider;
  bool uses_qdq_model = false;
  bool supports_multithread_inference = true;
  bool save_expected_outputs_to_disk = false;
  bool load_expected_outputs_from_disk = false;
  Ort::SessionOptions session_options;
};

bool ParseCmdLineArgs(AppArgs& app_args, int argc, char** argv);

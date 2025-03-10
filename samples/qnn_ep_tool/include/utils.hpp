// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem> //NOLINT
#include <string>
#include <vector>

#include "core/platform/path_lib.h"

#include "include/model_info.hpp"

std::basic_string<PATH_CHAR_TYPE> find_model_path(std::string model_dir);

std::vector<std::basic_string<PATH_CHAR_TYPE>> find_test_data_sets(std::string model_dir);

std::string check_data_format(const std::filesystem::path test_data_set_dir);

void load_input_tensors_from_raws(
  std::filesystem::path inp_dir,
  const OrtApi* g_ort,
  OnnxModelInfo* model_info,
  std::vector<std::vector<float>>* input_data
);

void dump_output_tensors_to_raws(
  std::filesystem::path out_dir,
  const OrtApi* g_ort,
  OnnxModelInfo* model_info
);

void load_input_tensors_from_pbs(
  std::filesystem::path inp_dir,
  const OrtApi* g_ort,
  OnnxModelInfo* model_info,
  std::vector<std::vector<float>>* input_data
);

void dump_output_tensors_to_pbs(
  std::filesystem::path out_dir,
  const OrtApi* g_ort,
  OnnxModelInfo* model_info
);
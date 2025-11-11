// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <filesystem>
#include <string>
#include <map>
#include <codecvt>

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_ep_device_ep_metadata_keys.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>
#include <core/providers/nv_tensorrt_rtx/nv_provider_options.h>

#include "core/graph/constants.h"
#include "core/common/path_string.h"
#include "core/framework/tensor.h"
#include "core/framework/ort_value.h"
#include "test/util/include/api_asserts.h"

//
// Note: This header file is copied from test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h
//       Some function declarations are removed as not needed.
//

namespace onnxruntime {
namespace test {

[[maybe_unused]] static std::string PathToUTF8(const PathString& path) {
#ifdef _WIN32
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(path);
#else
  return path.c_str();
#endif
}

[[maybe_unused]] static void clearFileIfExists(PathString path) {
  if (std::filesystem::exists(path)) {
    std::filesystem::remove(path);
  }
}

template <typename T>
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<T> found(rtensor.Data<T>(), rtensor.Data<T>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

/**
 * Create a simple model with dynamic or non-dynamic input shape.
 * \param model_name - model name
 * \param graph_name - graph name
 * \param dims - input dimensions
 * \param add_non_zero_node - add NonZero node which makes the whole model partition into TRT EP and CUDA EP subgraphs.
 *
 * input: "X", "Y" and "Z"
 *        you can specify input dimensions, for example (1, 3, 2), (1, 2) or (1, -1, -1)). Note: -1 means the dimension is dynamic.
 *        All three inputs have the same dimensions.
 * output: "M"
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *     "M"
 *
 *     or
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *    NonZero (This node will be placed on CUDA EP)
 *     /
 *   "M"
 */
void CreateBaseModel(const PathString& model_name,
                     std::string graph_name,
                     std::vector<int> dims,
                     bool add_non_zero_node = false);

void CreateLargeLLMModel(const PathString& model_path,
                         const PathString& external_data_path,
                         int num_layers = 32,
                         int hidden_dim = 2048);

Ort::IoBinding generate_io_binding(
    Ort::Session& session,
    std::map<std::string, std::vector<int64_t>> shape_overwrites = {},
    OrtAllocator* allocator = nullptr);

}  // namespace test
}  // namespace onnxruntime

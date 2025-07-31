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

#include "core/graph/constants.h"
#include "core/common/path_string.h"
#include "core/framework/tensor.h"
#include "core/framework/ort_value.h"
#include "test/util/include/api_asserts.h"

namespace onnxruntime {
namespace test {

using RegisteredEpDeviceUniquePtr = std::unique_ptr<const OrtEpDevice, std::function<void(const OrtEpDevice*)>>;

struct Utils {
  struct NvTensorRtRtxEpInfo {
    const std::filesystem::path library_path =
#if _WIN32
        "onnxruntime_providers_nv_tensorrt_rtx.dll";
#else
        "libonnxruntime_providers_nv_tensorrt_rtx.so";
#endif
    const std::string registration_name = kNvTensorRTRTXExecutionProvider;
  };

  static NvTensorRtRtxEpInfo nv_tensorrt_rtx_ep_info;

  // get the OrtEpDevice for the NV TensorRT RTX EP from the environment
  static void GetEp(Ort::Env& env, const std::string& ep_name, const OrtEpDevice*& ep_device);

  // Register the NV TensorRT RTX EP library, get the OrtEpDevice for it, and return a unique pointer that will
  // automatically unregister the EP library.
  static void RegisterAndGetNvTensorRtRtxEp(Ort::Env& env, RegisteredEpDeviceUniquePtr& nv_tensorrt_rtx_ep);
};

[[maybe_unused]] static std::string PathToUTF8(const PathString& path) {
#ifdef WIN32
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
static void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
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
 * \param add_fast_gelu - add FastGelu node which makes the whole model partition into TRT EP and CUDA EP subgraphs.
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
 *       Add (+ float scalar "S")
 *       /
 *     "O"
 *
 *     or
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *    FastGelu (This node will be placed on CUDA EP)
 *     /
 *     *       Add (+ float scalar "S")
 *    /
 *   "O"
 */
void CreateBaseModel(const PathString& model_name,
                     std::string graph_name,
                     std::vector<int> dims,
                     bool add_fast_gelu = false,
                     ONNX_NAMESPACE::TensorProto_DataType dtype = ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

Ort::IoBinding generate_io_binding(
    Ort::Session& session,
    std::map<std::string, std::vector<int64_t>> shape_overwrites = {},
    OrtAllocator* allocator = nullptr);

}  // namespace test
}  // namespace onnxruntime

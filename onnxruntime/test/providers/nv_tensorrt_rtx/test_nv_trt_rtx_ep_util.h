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

namespace onnxruntime {
namespace test {

using RegisteredEpDeviceUniquePtr = std::unique_ptr<const OrtEpDevice, std::function<void(const OrtEpDevice*)>>;

struct Utils {
  struct NvTensorRtRtxEpInfo {
    const std::filesystem::path library_path =
#ifdef _WIN32
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
 * \param external_initializer_file - file name to save external initializers to
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
                     ONNX_NAMESPACE::TensorProto_DataType dtype = ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                     const PathString& external_initializer_file = {});

void CreateLargeLLMModel(const PathString& model_path, const PathString& external_data_path);

Ort::IoBinding generate_io_binding(
    Ort::Session& session,
    std::map<std::string, std::vector<int64_t>> shape_overwrites = {},
    OrtAllocator* allocator = nullptr);

#if !defined(DISABLE_FLOAT8_TYPES)
/**
 * Create a model with TRT_FP8QuantizeLinear -> TRT_FP8DequantizeLinear nodes for per-tensor quantization.
 *
 * The model uses opset version 19 for ONNX domain and opset version 1 for TRT domain.
 *
 * \param model_name - output model file name
 * \param graph_name - name of the graph
 *
 * input: "X" [4, 64] (FLOAT16)
 * output: "Y" [4, 64] (FLOAT16)
 *
 *                "X" [4x64]
 *                 (FLOAT16)
 *                     |
 *                     |
 *            TRT_FP8QuantizeLinear (trt domain)
 *            (+ scale, FLOAT16 initializer)
 *                     |
 *                     |
 *              "X_quantized"
 *                  [4x64]
 *              (FLOAT8E4M3FN)
 *                     |
 *                     |
 *          TRT_FP8DequantizeLinear (trt domain)
 *            (+ scale, FLOAT16 initializer)
 *                     |
 *                     |
 *                   "Y"
 *                 [4x64]
 *                (FLOAT16)
 */
void CreateFP8CustomOpModel(const PathString& model_name,
                            const std::string& graph_name);
#endif  // !defined(DISABLE_FLOAT8_TYPES)

#if !defined(DISABLE_FLOAT4_TYPES) && !defined(DISABLE_FLOAT8_TYPES)
/**
 * Create a model with TRT_FP4DynamicQuantize node followed by DequantizeLinear nodes.
 *
 * The model uses opset version 23 for ONNX domain and opset version 1 for TRT domain.
 *
 * \param model_name - output model file name
 * \param graph_name - name of the graph
 *
 * input: "X" [64, 64] (FLOAT16)
 * output: "X_dequantized" [64, 64] (FLOAT16)
 *
 *                 "X" [64x64]
 *                  (FLOAT16)
 *                      |
 *                      |
 *              TRT_FP4DynamicQuantize (trt domain)
 *              (axis=-1, block_size=16, scale_type=17)
 *              (+ scale, FLOAT16 initializer)
 *                   /         \
 *                  /           \
 *     "X_quantized"             "X_scale"
 *     [64x64]                   [64x4]
 *     (FLOAT4E2M1)            (FLOAT8E4M3FN)
 *          |                        |
 *          |                        |
 *          |                  DequantizeLinear #1
 *          |                  (+ dequant_scale, FLOAT16 initializer)
 *          |                        |
 *          |                        |
 *          |                "X_scale_dequantized"
 *          |                      [64x4]
 *          |                     (FLOAT16)
 *          |                        /
 *           \                      /
 *            \                    /
 *             \                  /
 *              DequantizeLinear #2
 *           (axis=-1, block_size=16)
 *                     |
 *                     |
 *              "X_dequantized" [OUTPUT]
 *                  [64x64]
 *                 (FLOAT16)
 */
void CreateFP4CustomOpModel(const PathString& model_name,
                            const std::string& graph_name);
#endif  // !defined(DISABLE_FLOAT4_TYPES) && !defined(DISABLE_FLOAT8_TYPES)

}  // namespace test
}  // namespace onnxruntime

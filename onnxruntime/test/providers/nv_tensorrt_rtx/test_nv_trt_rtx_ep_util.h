// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#pragma once


#include <filesystem>
#include <string>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/constants.h"

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
}  // namespace test
}  // namespace onnxruntime

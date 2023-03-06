// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.h"

#ifdef __ANDROID__
#include <android/api-level.h>
#endif


namespace onnxruntime {
namespace nnapi {
struct NnApi;
class ModelBuilder;

enum class TargetDeviceOption : int8_t {
  ALL_DEVICES,  // use all available target devices

  /* TODO support these options
  PREFERRED_DEVICES,  // Use one or more preferred devices (must be given)
  EXCLUDED_DEVICES,   // Exclude one or more devices (must be given)
   */

  CPU_DISABLED,  // use all available target devices except CPU
  CPU_ONLY,      // use CPU only
};

const char* const nnapi_cpu = ("nnapi-reference");
int32_t GetNNAPIEffectiveFeatureLevel(const NnApi& nnapi_handle, const std::vector<ANeuralNetworksDevice*>& device_handles);

Status GetTargetDevices(const NnApi& nnapi_handle, TargetDeviceOption target_device_option,
                        std::vector<ANeuralNetworksDevice*>& nnapi_target_devices, std::string& nnapi_target_devices_detail);
int32_t GetNNAPIEffectiveFeatureLevelFromTargetDeviceOption(const NnApi& nnapi_handle, TargetDeviceOption target_device_option);
}  // namespace nnapi
}  // namespace onnxruntime

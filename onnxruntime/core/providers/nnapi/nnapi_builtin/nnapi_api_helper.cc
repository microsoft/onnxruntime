// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/nnapi_api_helper.h"

#include "core/common/inlined_containers_fwd.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "core/common/logging/logging.h"

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

namespace onnxruntime {
namespace nnapi {

static int32_t GetNNAPIRuntimeFeatureLevel(const NnApi& nnapi_handle) {
  auto runtime_level = static_cast<int32_t>(nnapi_handle.nnapi_runtime_feature_level);

#ifdef __ANDROID__
  int device_api_level = android_get_device_api_level();
  runtime_level = (device_api_level < __ANDROID_API_S__) ? device_api_level : runtime_level;
#endif
  return runtime_level;
}

/**
 * Get the max feature level supported by target devices. We want to run as many ops as possible
  on NNAPI devices except for nnapi-cpu. So we get the max feature level across all target devices.
 *
 * @param nnapi_handle nnapi-lib handle.
 * @param `devices` target devices users want to use.
 *
 * @return The max feature level across all devices or the runtime feature level if no devices are specified.
 *
 */
static int32_t GetDeviceFeatureLevelInternal(const NnApi& nnapi_handle, gsl::span<const DeviceWrapper> devices) {
  int32_t target_feature_level = GetNNAPIRuntimeFeatureLevel(nnapi_handle);

  int64_t devices_feature_level = -1;

  for (const auto& device : devices) {
    // We don't care which device runs them.
    devices_feature_level = std::max(device.feature_level, devices_feature_level);
  }

  // nnapi_cpu has the feature 1000
  if ((devices_feature_level > 0) && (devices_feature_level < target_feature_level)) {
    LOGS_DEFAULT(INFO) << "Changing NNAPI Feature Level " << target_feature_level
                       << " to supported by target devices: " << devices_feature_level;

    target_feature_level = static_cast<int32_t>(devices_feature_level);
  }
  return target_feature_level;
}

/**
 * Get all target devices specified by target_device_option
 * CPU device Will be put last
 *
 * @param nnapi_handle nnapi-lib handle.
 * @param target_device_option target devices users want to use.
 *
 * @return Status of the operation.
 *
 */
Status GetTargetDevices(const NnApi& nnapi_handle, TargetDeviceOption target_device_option,
                        DeviceWrapperVector& devices) {
  // GetTargetDevices is only supported when NNAPI runtime feature level >= ANEURALNETWORKS_FEATURE_LEVEL_3
  if (GetNNAPIRuntimeFeatureLevel(nnapi_handle) < ANEURALNETWORKS_FEATURE_LEVEL_3)
    return Status::OK();

  uint32_t num_devices = 0;
  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi_handle.ANeuralNetworks_getDeviceCount(&num_devices), "Getting count of available devices");

  int32_t cpu_index = -1;
  for (uint32_t i = 0; i < num_devices; i++) {
    ANeuralNetworksDevice* device = nullptr;
    const char* device_name = nullptr;
    int32_t device_type = 0;
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi_handle.ANeuralNetworks_getDevice(i, &device), "Getting " + std::to_string(i) + "th device");

    RETURN_STATUS_ON_ERROR_WITH_NOTE(nnapi_handle.ANeuralNetworksDevice_getName(device, &device_name),
                                     "Getting " + std::to_string(i) + "th device's name");

    RETURN_STATUS_ON_ERROR_WITH_NOTE(nnapi_handle.ANeuralNetworksDevice_getType(device, &device_type),
                                     "Getting " + std::to_string(i) + "th device's type");

    int64_t curr_device_feature_level = 0;
    RETURN_STATUS_ON_ERROR_WITH_NOTE(nnapi_handle.ANeuralNetworksDevice_getFeatureLevel(device, &curr_device_feature_level),
                                     "Getting " + std::to_string(i) + "th device's feature level");

    // https://developer.android.com/ndk/reference/group/neural-networks#aneuralnetworksdevice_gettype
    bool device_is_cpu = device_type == ANEURALNETWORKS_DEVICE_CPU;
    if ((target_device_option == TargetDeviceOption::CPU_DISABLED && device_is_cpu) ||
        (target_device_option == TargetDeviceOption::CPU_ONLY && !device_is_cpu)) {
      continue;
    }

    if (device_is_cpu) {
      cpu_index = static_cast<int32_t>(devices.size());
    }
    devices.push_back({device, std::string(device_name), device_type, curr_device_feature_level});
  }

  // Put CPU device at the end
  // 1) It's helpful to accelerate NNAPI compiling, just assuming nnapi-reference has the lowest priority
  // and it internally skips the last device if it has already found one.
  // 2) We can easily identify if the last device is nnapi-reference without traversing the whole device list.
  // and exclude nnapi-reference for better performance if possible.
  // 3) We can easily log the detail of how op was assigned on NNAPI devices which is helpful for debugging.
  // Related doc and code https://source.android.com/docs/core/interaction/neural-networks#cpu-usage
  // and https://android.googlesource.com/platform/frameworks/ml/+/5b525d4d9100819d87447bd2c2a0bcfdd62899ee/nn/runtime/ExecutionPlan.cpp#2303
  if (cpu_index != -1 && cpu_index != static_cast<int32_t>(devices.size()) - 1) {
    std::swap(devices[devices.size() - 1], devices[cpu_index]);
  }

  return Status::OK();
}

std::string GetDevicesDescription(gsl::span<const DeviceWrapper> devices) {
  std::string nnapi_target_devices_detail;
  for (const auto& device : devices) {
    const auto device_detail = MakeString("[Name: [", device.name, "], Type [", device.type, "]], ");
    nnapi_target_devices_detail += device_detail + " ,";
  }
  return nnapi_target_devices_detail;
}

// Get target devices first and then get the max feature level supported by target devices.
// Return -1 if failed.
// It's not necessary to handle the error here, because level=-1 will refuse all ops in NNAPI EP.
int32_t GetNNAPIEffectiveFeatureLevelFromTargetDeviceOption(const NnApi& nnapi_handle, TargetDeviceOption target_device_option) {
  DeviceWrapperVector nnapi_target_devices;
  if (auto st = GetTargetDevices(nnapi_handle, target_device_option, nnapi_target_devices); !st.IsOK()) {
    LOGS_DEFAULT(WARNING) << "GetTargetDevices failed for:" << st.ErrorMessage();
    return -1;
  }
  return GetDeviceFeatureLevelInternal(nnapi_handle, nnapi_target_devices);
}

// Get the max feature level supported by target devices.
// If no devices are specified, it will return the runtime feature level
int32_t GetNNAPIEffectiveFeatureLevel(const NnApi& nnapi_handle, gsl::span<const DeviceWrapper> device_handles) {
  return GetDeviceFeatureLevelInternal(nnapi_handle, device_handles);
}

}  // namespace nnapi
}  // namespace onnxruntime

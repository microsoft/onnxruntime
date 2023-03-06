// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nnapi_api_helper.h"

#include "core/common/inlined_containers_fwd.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "core/common/logging/logging.h"

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

namespace onnxruntime {
namespace nnapi {

/**  How feature level works for NNAPI. refer to https://developer.android.com/ndk/reference/group/neural-networks
 *
 * NNAPI device feature level is closely related to NNAPI runtime feature level
    (ANeuralNetworks_getRuntimeFeatureLevel), which indicates an NNAPI runtime feature level
    (the most advanced NNAPI specification and features that the runtime implements).
    An NNAPI device feature level is always less than or equal to the runtime feature level.
 *
 * On Android devices with API level 30 and older, the Android API level of the Android device
    must be used for NNAPI runtime feature discovery.
    Enum values in FeatureLevelCode from feature level 1 to 5 have their
    corresponding Android API levels listed in their documentation,
    and each such enum value equals the corresponding API level.
    This allows using the Android API level as the feature level.
    This mapping between enum value and Android API level does not exist for
    feature levels after NNAPI feature level 5 and API levels after S (31).

 */
static int32_t GetNNAPIRuntimeFeatureLevel(const NnApi& nnapi_handle) {
  int32_t runtime_level = static_cast<int32_t>(nnapi_handle.nnapi_runtime_feature_level);

#ifdef __ANDROID__
  int device_api_level = android_get_device_api_level();
  runtime_level = (device_api_level < __ANDROID_API_S__) ? device_api_level : runtime_level;
#endif
  return runtime_level;
}

/**
 * Get the max feature level supported by all target devices.
 *
 * @param nnapi_handle nnapi-lib handle.
 * @param device_handles target devices users want to use.
 *
 * @return The max feature level support by a set of devices.
 *
 */
static int32_t GetDeviceFeatureLevelInternal(const NnApi& nnapi_handle, const std::vector<DeviceWrapper>& device_sets) {
  int32_t target_feature_level = GetNNAPIRuntimeFeatureLevel(nnapi_handle);

  int64_t devices_feature_level = -1;

  for (const auto &device : device_sets) {
    // we want to op run on the device with the highest feature level so we can support more ops.
    // and we don't care which device runs them.
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

// get all target devices which satisfy the target_device_option
// we will always put CPU device at the end if cpu is enabled
Status GetTargetDevices(const NnApi& nnapi_handle, TargetDeviceOption target_device_option,
                        std::vector<DeviceWrapper>& device_sets) {
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
      cpu_index = static_cast<int32_t>(device_sets.size());
    }
    device_sets.push_back({device, std::string(device_name), device_type, curr_device_feature_level});
  }

  // put CPU device at the end
  // 1) it's helpful to accelerate nnapi compile, just assuming nnapi-reference has the lowest priority
  // and nnapi internally skip the last device if it has already found one.
  // 2) we can easily exclude nnapi-reference when not strict excluding CPU.
  // 3) we can easily log the detail of how op was assigned on NNAPI devices which is helpful for debugging.
  if (cpu_index != -1 && cpu_index != static_cast<int32_t>(device_sets.size()) - 1) {
    std::swap(device_sets[device_sets.size() - 1], device_sets[cpu_index]);
  }

  return Status::OK();
}


std::string GetDeviceDescription(const std::vector<DeviceWrapper>& device_sets) {
  std::string nnapi_target_devices_detail;
  for (const auto& device : device_sets) {
    const auto device_detail = MakeString("[Name: [", device.name, "], Type [", device.type, "]], ");
    nnapi_target_devices_detail += device_detail + " ,";
  }
  return nnapi_target_devices_detail;
}

// Get devices-set first and then get the max feature level supported by all target devices
// return -1 if failed.  It's not necessary to handle the error here, because level=-1 will refuse all ops
int32_t GetNNAPIEffectiveFeatureLevelFromTargetDeviceOption(const NnApi& nnapi_handle, TargetDeviceOption target_device_option) {
  std::vector<DeviceWrapper> nnapi_target_devices;
  if (auto st = GetTargetDevices(nnapi_handle, target_device_option, nnapi_target_devices); !st.IsOK()) {
    LOGS_DEFAULT(WARNING) << "GetTargetDevices failed for :" << st.ErrorMessage();
    return -1;
  }
  return GetDeviceFeatureLevelInternal(nnapi_handle, nnapi_target_devices);
}

// get the max feature level supported by all target devices, If no devices are specified,
// it will return the runtime feature level
int32_t GetNNAPIEffectiveFeatureLevel(const NnApi& nnapi_handle, const std::vector<DeviceWrapper>& device_handles) {
  return GetDeviceFeatureLevelInternal(nnapi_handle, device_handles);
}

}  // namespace nnapi
}  // namespace onnxruntime

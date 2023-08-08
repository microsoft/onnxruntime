// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.h"

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

struct NnApi;
namespace onnxruntime {
namespace nnapi {

class ModelBuilder;

struct DeviceWrapper {
  ANeuralNetworksDevice* device;
  std::string name;
  int32_t type;
  int64_t feature_level;
};

enum class TargetDeviceOption : int8_t {
  ALL_DEVICES,  // use all available target devices

  /* TODO support these options
  PREFERRED_DEVICES,  // Use one or more preferred devices (must be given)
  EXCLUDED_DEVICES,   // Exclude one or more devices (must be given)
   */

  CPU_DISABLED,  // use all available target devices except CPU
  CPU_ONLY,      // use CPU only
};

constexpr const char* const kNnapiCpuDeviceName = "nnapi-reference";

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
int32_t GetNNAPIEffectiveFeatureLevel(const NnApi& nnapi_handle, gsl::span<const DeviceWrapper> device_handles);

using DeviceWrapperVector = InlinedVector<DeviceWrapper>;

/**
 * Get all target devices specified by target_device_option.
 */
Status GetTargetDevices(const NnApi& nnapi_handle, TargetDeviceOption target_device_option,
                        DeviceWrapperVector& nnapi_target_devices);

int32_t GetNNAPIEffectiveFeatureLevelFromTargetDeviceOption(const NnApi& nnapi_handle, TargetDeviceOption target_device_option);

std::string GetDevicesDescription(gsl::span<const DeviceWrapper> devices);
}  // namespace nnapi
}  // namespace onnxruntime

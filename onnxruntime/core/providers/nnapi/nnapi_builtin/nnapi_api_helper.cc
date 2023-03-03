#include "nnapi_api_helper.h"

#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "core/common/logging/logging.h"

#ifdef __ANDROID__
#include <android/api-level.h>
#endif

namespace onnxruntime {
namespace nnapi {

int32_t GetNNAPIRuntimeFeatureLevel(const ::NnApi* nnapi) {
  if (!nnapi)
    return 0;
  int32_t runtime_level = static_cast<int32_t>(nnapi->nnapi_runtime_feature_level);
#ifdef __ANDROID__
  int device_api_level = android_get_device_api_level();
  runtime_level = (device_api_level < __ANDROID_API_S__) ? device_api_level : runtime_level;
#endif
  return runtime_level;
}

int32_t GetDeviceFeatureLevel(const ::NnApi* nnapi, const std::vector<ANeuralNetworksDevice*>& device_handles) {
  if (!nnapi)
    return 0;

  int32_t target_feature_level = GetNNAPIRuntimeFeatureLevel(nnapi);

  int64_t devices_feature_level = -1;

  for (const auto* device_handle : device_handles) {
    int64_t curr_device_feature_level = 0;
    if (nnapi->ANeuralNetworksDevice_getFeatureLevel(device_handle, &curr_device_feature_level) != ANEURALNETWORKS_NO_ERROR) {
      continue;
    }

    devices_feature_level = std::max(curr_device_feature_level, devices_feature_level);
  }

  if ((devices_feature_level > 0) && (devices_feature_level < target_feature_level)) {
    LOGS_DEFAULT(INFO) << "Changing NNAPI Feature Level " << target_feature_level
                       << " to supported by target devices: " << devices_feature_level;

    target_feature_level = static_cast<int32_t>(devices_feature_level);
  }
  return target_feature_level;
}

Status GetTargetDevices(const ::NnApi* nnapi, TargetDeviceOption target_device_option,
                                      std::vector<ANeuralNetworksDevice*>& nnapi_target_devices, std::string& nnapi_target_devices_detail) {
  // GetTargetDevices is only supported on API 29+
  // get runtime_feature_level
  if (GetNNAPIRuntimeFeatureLevel(nnapi) < ANEURALNETWORKS_FEATURE_LEVEL_3)
    return Status::OK();

  const std::string nnapi_cpu("nnapi-reference");
  uint32_t num_devices = 0;
  RETURN_STATUS_ON_ERROR_WITH_NOTE(
      nnapi->ANeuralNetworks_getDeviceCount(&num_devices), "Getting count of available devices");

  int32_t cpu_index = -1;
  for (uint32_t i = 0; i < num_devices; i++) {
    ANeuralNetworksDevice* device = nullptr;
    const char* device_name = nullptr;
    int32_t device_type = 0;
    RETURN_STATUS_ON_ERROR_WITH_NOTE(
        nnapi->ANeuralNetworks_getDevice(i, &device), "Getting " + std::to_string(i) + "th device");

    RETURN_STATUS_ON_ERROR_WITH_NOTE(nnapi->ANeuralNetworksDevice_getName(device, &device_name),
                                     "Getting " + std::to_string(i) + "th device's name");

    RETURN_STATUS_ON_ERROR_WITH_NOTE(nnapi->ANeuralNetworksDevice_getType(device, &device_type),
                                     "Getting " + std::to_string(i) + "th device's type");
    bool device_is_cpu = nnapi_cpu == device_name;
    if ((target_device_option == TargetDeviceOption::CPU_DISABLED) && device_is_cpu) {
      continue;
    }

    if (device_is_cpu) {
      cpu_index = int32_t(nnapi_target_devices.size());
    }
    nnapi_target_devices.push_back(device);
    const auto device_detail = MakeString("[Name: [", device_name, "], Type [", device_type, "]], ");
    nnapi_target_devices_detail += device_detail;
  }

  // put CPU device at the end
  if (cpu_index != -1 && cpu_index != int32_t(nnapi_target_devices.size()) - 1) {
    std::swap(nnapi_target_devices[nnapi_target_devices.size() - 1], nnapi_target_devices[cpu_index]);
  }

  return Status::OK();
}

int32_t GetNNAPIFeatureLevelWithDeviceTag(const ::NnApi* nnapi, TargetDeviceOption target_device_option) {
  std::vector<ANeuralNetworksDevice*> nnapi_target_devices;
  std::string nnapi_target_devices_detail;
  if (!GetTargetDevices(nnapi, target_device_option, nnapi_target_devices, nnapi_target_devices_detail).IsOK()) {
    LOGS_DEFAULT(WARNING) << "GetTargetDevices failed";
  }
  LOGS_DEFAULT(VERBOSE) << "finding devices [" << nnapi_target_devices_detail << "] in NNAPI";
  return GetDeviceFeatureLevel(nnapi, nnapi_target_devices);
}

int32_t GetNNAPIFeatureLevel(const ModelBuilder& model_builder) {
  return GetDeviceFeatureLevel(model_builder.GetNnApi(), model_builder.GetDevices());
}

int32_t GetNNAPIFeatureLevel(const ::NnApi* nnapi, const std::vector<ANeuralNetworksDevice*>& device_handles) {
  if (!nnapi)
    return 0;

  return GetDeviceFeatureLevel(nnapi, device_handles);
}

}  // namespace nnapi
}  // namespace onnxruntime

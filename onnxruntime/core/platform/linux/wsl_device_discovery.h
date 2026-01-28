#pragma once

#include "core/platform/device_discovery.h"
#include "core/common/common.h"
#include <vector>

namespace onnxruntime {
Status DetectGpuNvml(std::vector<OrtHardwareDevice>& gpu_devices_out);
Status
DetectGpuIfWsl(std::vector<OrtHardwareDevice>& gpu_devices_out);
}  // namespace onnxruntime

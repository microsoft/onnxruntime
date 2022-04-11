// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "opencl_execution_provider.h"


namespace onnxruntime {
namespace opencl {
std::vector<size_t> AdrenoLocalSize2D(const opencl::NDRange& gws, const opencl::OpenCLDeviceInfo& gpu_info);
opencl::NDRange RunTuneLWS2D(const opencl::NDRange& gws, opencl::OpenCLDeviceInfo dev_info_, const opencl::TuneKernelWithTimeFunc& func, int32_t auto_tuning_level);
}
}  // namespace onnxruntime

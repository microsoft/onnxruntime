// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_COMPOSABLE_KERNEL
#include "contrib_ops/rocm/diffusion/group_norm_ck_impl/impl.cuh"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_impl.hpp"

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace internal {

template <>
std::vector<std::unique_ptr<DeviceNormalization<F16, F32, F32, F32, F16, Swish, 5, 3>>>
GetDeviceGroupNormInstances<F16, F32, F32, F32, F16, Swish, 5, 3>() {
  std::vector<std::unique_ptr<DeviceNormalization<F16, F32, F32, F32, F16, Swish, 5, 3>>> instances;
  ck::tensor_operation::device::instance::add_device_operation_instances(
      instances,
      device_normalization_f16_instances<Swish, 5, 3>{});

  return instances;
}

template <>
std::vector<std::unique_ptr<DeviceNormalization<F16, F32, F32, F32, F16, Pass, 5, 3>>>
GetDeviceGroupNormInstances<F16, F32, F32, F32, F16, Pass, 5, 3>() {
  std::vector<std::unique_ptr<DeviceNormalization<F16, F32, F32, F32, F16, Pass, 5, 3>>> instances;
  ck::tensor_operation::device::instance::add_device_operation_instances(
      instances,
      device_normalization_f16_instances<Pass, 5, 3>{});

  return instances;
}

}  // namespace internal
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
#endif  // USE_COMPOSABLE_KERNEL

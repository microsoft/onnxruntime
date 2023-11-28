// SPDX-License-Identifier: MIT
// Modifications Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifdef USE_COMPOSABLE_KERNEL
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization_fwd.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_fwd_impl.hpp"
#include "ck/utility/data_type.hpp"

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace internal {

using F16 = ck::half_t;
using F32 = float;

using Swish = ck::tensor_operation::element_wise::Swish;
using Pass = ck::tensor_operation::element_wise::PassThrough;

using ck::tensor_operation::device::DeviceNormalizationFwd;      // the interface
using ck::tensor_operation::device::DeviceNormalizationFwdImpl;  // the implementation

// See https://github.com/ROCmSoftwarePlatform/composable_kernel/blob/1fefd82ed8/library/src/tensor_operation_instance/gpu/normalization_fwd/normalization_fwd_instance_common.hpp

template <typename OutElementwise, ck::index_t Rank, ck::index_t Reduce>
using device_normalization_f32_instances = std::tuple<
    // clang-format off
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, SaveMeanInvStdDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorDim, GammaSrcVectorSize, BetaSrcVectorDim, BetaSrcVectorSize, YDstVectorSize, SaveMeanInvStdScalarPerVector>
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, // irregular size
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1>,   // irregular size
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 4, 1, 4, 1, 4, 4, 2>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 2, 8, 1, 4, 1, 4, 1, 4, 4, 2>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,
        DeviceNormalizationFwdImpl<F32, F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 4, 1, 4, 1, 4, 4, 1>
    // clang-format on
    >;

template <typename OutElementwise, ck::index_t Rank, ck::index_t Reduce>
using device_normalization_f16_instances =
    // clang-format off
    std::tuple <
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, SaveMeanInvStdDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorDim, GammaSrcVectorSize, BetaSrcVectorDim, BetaSrcVectorSize, YDstVectorSize, SaveMeanInvStdScalarPerVector>
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>, // irregular size
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1>,   // irregular size
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4, 1>,   // irregular size
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 64, 1, 64, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 8, 1, 8, 1, 8, 8, 1>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 8, 1, 8, 1, 8, 8, 2>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 8, 1, 8, 1, 8, 8, 1>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 8, 1, 8, 1, 8, 8, 1>,
        DeviceNormalizationFwdImpl<F16, F32, F32, F32, F16, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 16, 1, 8, 1, 8, 1, 8, 8, 1>
    // clang-format on
    >;

// Use this function to get implementation
template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename SaveMeanInvStdDataType,
          typename YElementwiseOperation,
          ck::index_t Rank,
          ck::index_t NumReduceDim>
std::vector<std::unique_ptr<DeviceNormalizationFwd<XDataType,
                                                   GammaDataType,
                                                   BetaDataType,
                                                   YDataType,
                                                   SaveMeanInvStdDataType,
                                                   YElementwiseOperation,
                                                   Rank,
                                                   NumReduceDim>>>
GetDeviceGroupNormInstances() {
  return {};
}

template <>
std::vector<std::unique_ptr<DeviceNormalizationFwd<
    F16, F32, F32, F16, F32, Swish, 5, 3>>>
GetDeviceGroupNormInstances<
    F16, F32, F32, F16, F32, Swish, 5, 3>();

template <>
std::vector<std::unique_ptr<DeviceNormalizationFwd<
    F16, F32, F32, F16, F32, Pass, 5, 3>>>
GetDeviceGroupNormInstances<
    F16, F32, F32, F16, F32, Pass, 5, 3>();

template <>
std::vector<std::unique_ptr<DeviceNormalizationFwd<
    F32, F32, F32, F32, F32, Swish, 5, 3>>>
GetDeviceGroupNormInstances<
    F32, F32, F32, F32, F32, Swish, 5, 3>();

template <>
std::vector<std::unique_ptr<DeviceNormalizationFwd<
    F32, F32, F32, F32, F32, Pass, 5, 3>>>
GetDeviceGroupNormInstances<
    F32, F32, F32, F32, F32, Pass, 5, 3>();

}  // namespace internal
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
#endif  // USE_COMPOSABLE_KERNEL

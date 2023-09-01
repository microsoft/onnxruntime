// SPDX-License-Identifier: MIT
// Modifications Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifdef USE_COMPOSABLE_KERNEL
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_normalization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_impl.hpp"
#include "ck/utility/data_type.hpp"

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace internal {

using F16 = ck::half_t;
using F32 = float;

using Swish = ck::tensor_operation::element_wise::Swish;
using Pass = ck::tensor_operation::element_wise::PassThrough;

using ck::tensor_operation::device::DeviceNormalization;      // the interface
using ck::tensor_operation::device::DeviceNormalizationImpl;  // the implementation

template <typename OutElementwise, ck::index_t Rank, ck::index_t Reduce>
using device_normalization_f32_instances = std::tuple<
    // clang-format off
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, OutElementwise, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorSize, BetaSrcVectorSize, YDstVectorSize>
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1>,   // irregular size
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1>, // irregular size
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2>,   // irregular size
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 512, 1, 512, 2, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F32, F32, F32, F32, F32, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 4, 1, 4, 1, 4, 4>
    // clang-format on
    >;

template <typename OutElementwise, ck::index_t Rank, ck::index_t Reduce>
using device_normalization_f16_instances = std::tuple<
    // clang-format off
        // XDataType, GammaDataType, BetaDataType, ComputeDataType, YDataType, OutElementwise, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, XYSrcVectorDim, XSrcVectorSize, GammaSrcVectorSize, BetaSrcVectorSize, YDstVectorSize>
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 1, 1, 1, 1, 1, 1, 1, 1>,    // irregular size
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 1, 1, 1, 1, 1, 1, 1, 1>,    // irregular size
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 1, 1, 1, 1, 1, 1, 1, 1>,    // irregular size
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 1, 1, 1, 1, 1, 1, 1, 1>,  // irregular size
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 2, 1, 2, 1, 2, 1, 2, 2>,    // irregular size
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 128, 1, 128, 1, 32, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 2, 16, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 256, 1, 256, 1, 32, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 1, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 512, 1, 512, 2, 8, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 4, 1, 4, 1, 4, 1, 4, 4>,
        DeviceNormalizationImpl<F16, F32, F32, F32, F16, OutElementwise, Rank, Reduce, 1024, 1, 1024, 1, 8, 1, 4, 1, 4, 1, 4, 4>
    // clang-format on
    >;

// Use this function to get implementation
template <typename InDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename OutDataType,
          typename YElementwiseOperation,
          ck::index_t Rank,
          ck::index_t NumReduceDim>
std::vector<std::unique_ptr<DeviceNormalization<InDataType,
                                                GammaDataType,
                                                BetaDataType,
                                                AccDataType,
                                                OutDataType,
                                                YElementwiseOperation,
                                                Rank,
                                                NumReduceDim>>>
GetDeviceGroupNormInstances() {
  return {};
}

template <>
std::vector<std::unique_ptr<DeviceNormalization<
    F16, F32, F32, F32, F16, Swish, 5, 3>>>
GetDeviceGroupNormInstances<
    F16, F32, F32, F32, F16, Swish, 5, 3>();

template <>
std::vector<std::unique_ptr<DeviceNormalization<
    F16, F32, F32, F32, F16, Pass, 5, 3>>>
GetDeviceGroupNormInstances<
    F16, F32, F32, F32, F16, Pass, 5, 3>();

template <>
std::vector<std::unique_ptr<DeviceNormalization<
    F32, F32, F32, F32, F32, Swish, 5, 3>>>
GetDeviceGroupNormInstances<
    F32, F32, F32, F32, F32, Swish, 5, 3>();

template <>
std::vector<std::unique_ptr<DeviceNormalization<
    F32, F32, F32, F32, F32, Pass, 5, 3>>>
GetDeviceGroupNormInstances<
    F32, F32, F32, F32, F32, Pass, 5, 3>();

}  // namespace internal
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
#endif  // USE_COMPOSABLE_KERNEL

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_COMPOSABLE_KERNEL
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_ck_impl/impl.cuh"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace internal {

using NonBiasedNonmasked = DeviceBatchedGemmSoftmaxGemmPermute<
    2, 1, 1, 1, 1,
    F16, F16, F16, F16, ck::Tuple<>, ck::Tuple<>,
    PassThrough, PassThrough, PreSoftmaxAttentionScoreOp, PassThrough, PassThrough,
    MaskingSpecialization::MaskDisabled>;

template <>
std::vector<std::unique_ptr<NonBiasedNonmasked>>
GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
    F16, ck::Tuple<>, F32, PreSoftmaxAttentionScoreOp, MaskingSpecialization::MaskDisabled>() {
  std::vector<std::unique_ptr<NonBiasedNonmasked>> instances;
  ck::tensor_operation::device::instance::add_device_operation_instances(
      instances,
      device_batched_gemm_softmax_gemm_permute_instances<
          2, 1, 1, 1, 1,
          F16, ck::Tuple<>, F32, PreSoftmaxAttentionScoreOp,
          MaskingSpecialization::MaskDisabled>{});

  return instances;
}

using NonBiasedNonmaskedCausal = DeviceBatchedGemmSoftmaxGemmPermute<
    2, 1, 1, 1, 1,
    F16, F16, F16, F16, ck::Tuple<>, ck::Tuple<>,
    PassThrough, PassThrough, PreSoftmaxAttentionScoreOp, PassThrough, PassThrough,
    MaskingSpecialization::MaskOutUpperTriangle>;

template <>
std::vector<std::unique_ptr<NonBiasedNonmaskedCausal>>
GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
    F16, ck::Tuple<>, F32, PreSoftmaxAttentionScoreOp, MaskingSpecialization::MaskOutUpperTriangle>() {
  std::vector<std::unique_ptr<NonBiasedNonmaskedCausal>> instances;
  ck::tensor_operation::device::instance::add_device_operation_instances(
      instances,
      device_batched_gemm_softmax_gemm_permute_instances<
          2, 1, 1, 1, 1,
          F16, ck::Tuple<>, F32, PreSoftmaxAttentionScoreOp,
          MaskingSpecialization::MaskOutUpperTriangle>{});

  return instances;
}

}  // namespace internal
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
#endif  // USE_COMPOSABLE_KERNEL

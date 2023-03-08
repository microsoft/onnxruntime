// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if 0  // Disable due to low perf with int type elementwise op
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_ck_impl/impl.cuh"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace internal {

using NonbiasedMasked = DeviceBatchedGemmSoftmaxGemmPermute<
    2, 1, 1, 1, 1,
    F16, F16, F16, F16, ck::Tuple<int32_t>, ck::Tuple<>,
    PassThrough, PassThrough, PreSoftmaxAttentionScoreOp, PassThrough, PassThrough,
    MaskingSpecialization::MaskDisabled>;

template <>
std::vector<std::unique_ptr<NonbiasedMasked>>
GetDeviceBatchedGemmSoftmaxGemmPermuteInstances<
    F16, ck::Tuple<int32_t>, F32, PreSoftmaxAttentionScoreOp, MaskingSpecialization::MaskDisabled>() {
  std::vector<std::unique_ptr<NonbiasedMasked>> instances;
  ck::tensor_operation::device::instance::add_device_operation_instances(
      instances,
      device_batched_gemm_softmax_gemm_permute_instances<
          2, 1, 1, 1, 1,
          F16, ck::Tuple<int32_t>, F32, PreSoftmaxAttentionScoreOp,
          MaskingSpecialization::MaskDisabled>{});

  return instances;
}

}  // namespace internal
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
#endif

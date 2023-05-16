// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/ck_attention_bias_mask_preprocess.cuh"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_impl.hpp"

#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"

namespace onnxruntime {
namespace contrib {
namespace rocm {
namespace internal {

// struct ConvertMask {
//   ConvertMask(float mask_filter_value) : mask_filter_value_(mask_filter_value) {}

//   template <typename BiasType, typename MaskType>
//   __host__ __device__ void operator()(BiasType& y, const MaskType& mask) const {
//     float filter_value = mask == 1 ? 0.0f : mask_filter_value_;
//     y = ck::type_convert<BiasType>(filter_value);
//   }

//   float mask_filter_value_;
// };

// Status LaunchConvertMask(const GemmSoftmaxGemmPermuteParams<half>* params) {
//   using BiasType = half;
//   using MaskType = int32_t;
//   constexpr const int NumDim = 3;
//   constexpr const int MPerThread = 8;

//   using Instance = ck::tensor_operation::device::DeviceElementwiseImpl<
//       ck::Tuple<MaskType>,
//       ck::Tuple<BiasType>,
//       ConvertMask,
//       NumDim,
//       MPerThread,
//       ck::Sequence<8>,
//       ck::Sequence<8>>;

//   auto [buffer, sizes, strides] = GetRawMaskBufferAddrSizesAndStrides(params->mask_index_buffer, params->attention);

//   auto impl = Instance{};
//   auto arg = impl.MakeArgumentPointer(
//       {sizes.x, sizes.y, sizes.z},
//       {{strides.x, strides.y, strides.z}},
//       {{strides.x, strides.y, strides.z}},
//       {buffer},
//       {params->workspace_buffer},
//       ConvertMask{params->attention->mask_filter_value});

//   auto invoker = impl.MakeInvokerPointer();
//   ORT_ENFORCE(impl.IsSupportedArgument(arg));
//   invoker->Run(arg, {params->Stream()});
//   return Status::OK();
// }

struct AddBiasAndConvertMask {
  AddBiasAndConvertMask(float mask_filter_value) : mask_filter_value_(mask_filter_value) {}

  template <typename BiasType, typename MaskType>
  __host__ __device__ void operator()(BiasType& y, const BiasType& bias, const MaskType& mask) const {
    float filter_value = mask == 1 ? 0.0f : mask_filter_value_;
    y = bias + ck::type_convert<BiasType>(filter_value);
  }

  float mask_filter_value_;
};

template <typename BiasType, typename MaskType>
using AddBiasAndConvertMaskInstance =
    ck::tensor_operation::device::DeviceElementwiseImpl<ck::Tuple<BiasType, MaskType>,
                                                        ck::Tuple<BiasType>,
                                                        AddBiasAndConvertMask,
                                                        /*NumDim*/ 4,
                                                        /*MPerThread*/ 8,
                                                        ck::Sequence<8, 8>,
                                                        ck::Sequence<8>>;

Status LaunchAddBiasAndConvertMask(const GemmSoftmaxGemmPermuteParams<half>* params) {
  using BiasType = ck::half_t;
  using MaskType = int32_t;
  constexpr const int NumDim = 4;
  constexpr const int MPerThread = 8;

  using Instance = ck::tensor_operation::device::DeviceElementwiseImpl<
      ck::Tuple<BiasType, MaskType>,
      ck::Tuple<BiasType>,
      AddBiasAndConvertMask,
      NumDim,
      MPerThread,
      ck::Sequence<8, 8>,
      ck::Sequence<8>>;

  int B = params->attention->batch_size;
  int N = params->attention->num_heads;
  int S = params->attention->sequence_length;
  int T = params->attention->total_sequence_length;
  std::array<ck::index_t, 4> bias_sizes = {B, N, S, T};
  std::array<ck::index_t, 4> bias_strides = {N * S * T, S * T, T, 1};

  auto [mask_buffer, mask_sizes, mask_strides] = GetRawMaskBufferAddrSizesAndStrides(
      params->mask_index_buffer, params->attention);

  ORT_ENFORCE(mask_sizes.x == B && (mask_sizes.y == 1 || mask_sizes.y == S) && mask_sizes.z == T);

  auto impl = Instance{};
  auto arg = impl.MakeArgumentPointer(
      bias_sizes,
      {bias_strides, {mask_strides.x, 0, mask_strides.y, mask_strides.z}},
      {bias_strides},
      {params->bias_buffer, mask_buffer},
      {params->workspace_buffer},
      AddBiasAndConvertMask{params->attention->mask_filter_value});

  auto invoker = impl.MakeInvokerPointer();
  ORT_ENFORCE(impl.IsSupportedArgument(arg.get()));
  invoker->Run(arg.get(), {params->Stream()});
  return Status::OK();
}

}  // namespace internal
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime

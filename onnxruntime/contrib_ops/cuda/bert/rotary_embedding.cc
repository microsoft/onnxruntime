// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/rotary_embedding_helper.h"
#include "contrib_ops/cuda/bert/rotary_embedding.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"
#include "core/providers/cuda/cuda_type_conversion.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::rotary_embedding_helper;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T, Version)                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      RotaryEmbedding,                                                  \
      kMSDomain,                                                        \
      Version,                                                          \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>())  \
          .TypeConstraint("TI", DataTypeImpl::GetTensorType<int32_t>()) \
          .InputMemoryType(OrtMemTypeCPUInput, 4),                      \
      RotaryEmbedding<T>);

REGISTER_KERNEL_TYPED(float, 1)
REGISTER_KERNEL_TYPED(MLFloat16, 1)
REGISTER_KERNEL_TYPED(BFloat16, 1)
REGISTER_KERNEL_TYPED(float, 2)
REGISTER_KERNEL_TYPED(MLFloat16, 2)
REGISTER_KERNEL_TYPED(BFloat16, 2)

template <typename T>
RotaryEmbedding<T>::RotaryEmbedding(const OpKernelInfo& info) : CudaKernel(info) {
  scale = info.GetAttrOrDefault<float>("scale", 1.0);
  rotary_embedding_dim = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0));
  num_heads = static_cast<int>(info.GetAttrOrDefault<int64_t>("num_heads", 0));
  interleaved = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);
  is_packed_batching = (info.GetAttrOrDefault<int64_t>("is_packed_batching", 0) == 1);
  const auto mode = info.GetAttrOrDefault<int64_t>("use_fp32_arithmetic", 0);
  ORT_ENFORCE(mode == 0 || mode == 1, "use_fp32_arithmetic must be 0 or 1");
  use_fp32_arithmetic = mode == 1;
}

template <typename T>
Status RotaryEmbedding<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* position_ids = context->Input<Tensor>(1);
  const Tensor* cos_cache = context->Input<Tensor>(2);
  const Tensor* sin_cache = context->Input<Tensor>(3);
  const Tensor* numerics = context->Input<Tensor>(4);
  ORT_RETURN_IF_NOT(numerics == nullptr ||
                        (numerics->IsDataType<int32_t>() && numerics->Shape().Size() == 1 &&
                         (numerics->Data<int32_t>()[0] == 0 || numerics->Data<int32_t>()[0] == 1)),
                    "prefill_numerics must be an int32 scalar containing 0 or 1");
  const bool dense_prefill = numerics != nullptr && numerics->Data<int32_t>()[0] == 1;

  RotaryParameters parameters = {};
  ORT_RETURN_IF_ERROR(rotary_embedding_helper::CheckInputs<Tensor>(input,
                                                                   position_ids,
                                                                   cos_cache,
                                                                   sin_cache,
                                                                   num_heads,
                                                                   rotary_embedding_dim,
                                                                   &parameters));

  Tensor* output = context->Output(0, input->Shape());

  if (is_packed_batching == false && parameters.sequence_length > parameters.max_sequence_length) {
    // Launch update_cos_sin_cache kernel with scale
    ORT_NOT_IMPLEMENTED("Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported");
  }

  // Launch rotary embedding kernel
  typedef typename ToCudaType<T>::MappedType CudaT;
  auto& device_prop = GetDeviceProp();
  const auto launch = [&](auto* type_tag) {
    using KernelT = std::remove_pointer_t<decltype(type_tag)>;
    return LaunchRotaryEmbeddingKernel<KernelT>(
      Stream(context),
      reinterpret_cast<KernelT*>(output->template MutableData<T>()),
      reinterpret_cast<const KernelT*>(input->template Data<T>()),
      position_ids->Data<int64_t>(),
      nullptr,  // past_sequence_lengths
      reinterpret_cast<const KernelT*>(cos_cache->template Data<T>()),
      reinterpret_cast<const KernelT*>(sin_cache->template Data<T>()),
      parameters.batch_size,
      parameters.sequence_length,
      parameters.num_heads,
      parameters.head_size,
      parameters.rotary_embedding_dim,
      parameters.max_sequence_length,
      parameters.position_ids_format,
      interleaved,
      device_prop.maxThreadsPerBlock,
      parameters.transposed,
      dense_prefill ? false : use_fp32_arithmetic);
  };
  if (dense_prefill) {
    return launch(static_cast<typename OrtToCudaType<T>::type*>(nullptr));
  }
  return launch(static_cast<CudaT*>(nullptr));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

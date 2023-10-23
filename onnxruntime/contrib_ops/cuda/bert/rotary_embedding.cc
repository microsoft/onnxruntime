// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/rotary_embedding_helper.h"
#include "contrib_ops/cuda/bert/rotary_embedding.h"
#include "contrib_ops/cuda/bert/rotary_embedding_impl.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::rotary_embedding_helper;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      RotaryEmbedding,                                                  \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()), \
      RotaryEmbedding<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
RotaryEmbedding<T>::RotaryEmbedding(const OpKernelInfo& info) : CudaKernel(info) {
  scale = info.GetAttrOrDefault<float>("scale", 1.0);
  interleaved = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);
}

template <typename T>
Status RotaryEmbedding<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* position_ids = context->Input<Tensor>(1);
  const Tensor* cos_cache = context->Input<Tensor>(2);
  const Tensor* sin_cache = context->Input<Tensor>(3);

  RotaryParameters parameters = {};
  ORT_RETURN_IF_ERROR(rotary_embedding_helper::CheckInputs<Tensor>(input,
                                                                   position_ids,
                                                                   cos_cache,
                                                                   sin_cache,
                                                                   &parameters));

  Tensor* output = context->Output(0, input->Shape());

  if (parameters.sequence_length > parameters.max_sequence_length) {
    // Launch update_cos_sin_cache kernel with scale
    ORT_NOT_IMPLEMENTED("Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported");
  }

  // Launch rotary embedding kernel
  typedef typename ToCudaType<T>::MappedType CudaT;
  auto& device_prop = GetDeviceProp();
  return LaunchRotaryEmbeddingKernel<CudaT>(
      Stream(context),
      reinterpret_cast<CudaT*>(output->template MutableData<T>()),
      reinterpret_cast<const CudaT*>(input->template Data<T>()),
      position_ids->Data<int64_t>(),
      reinterpret_cast<const CudaT*>(cos_cache->template Data<T>()),
      reinterpret_cast<const CudaT*>(sin_cache->template Data<T>()),
      parameters.batch_size,
      parameters.sequence_length,
      parameters.num_heads,
      parameters.head_size,
      parameters.max_sequence_length,
      parameters.position_ids_format,
      interleaved,
      device_prop.maxThreadsPerBlock);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

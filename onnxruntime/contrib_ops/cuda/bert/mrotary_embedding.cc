// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/mrotary_embedding_helper.h"
#include "contrib_ops/cuda/bert/mrotary_embedding.h"
#include "contrib_ops/cuda/bert/mrotary_embedding_impl.h"

#include <limits>

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::mrotary_embedding_helper;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      MRotaryEmbedding,                                                 \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()), \
      MRotaryEmbedding<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
MRotaryEmbedding<T>::MRotaryEmbedding(const OpKernelInfo& info) : CudaKernel(info) {
  scale = info.GetAttrOrDefault<float>("scale", 1.0);
  const int64_t rotary_embedding_dim_attr = info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0);
  const int64_t num_heads_attr = info.GetAttrOrDefault<int64_t>("num_heads", 0);
  ORT_ENFORCE(rotary_embedding_dim_attr >= 0 && rotary_embedding_dim_attr <= std::numeric_limits<int>::max(),
              "rotary_embedding_dim must be in range [0, ", std::numeric_limits<int>::max(),
              "]. Actual value: ", rotary_embedding_dim_attr);
  ORT_ENFORCE(num_heads_attr >= 0 && num_heads_attr <= std::numeric_limits<int>::max(),
              "num_heads must be in range [0, ", std::numeric_limits<int>::max(),
              "]. Actual value: ", num_heads_attr);
  rotary_embedding_dim = static_cast<int>(rotary_embedding_dim_attr);
  num_heads = static_cast<int>(num_heads_attr);
  interleaved = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);
  is_packed_batching = (info.GetAttrOrDefault<int64_t>("is_packed_batching", 0) == 1);
  mrope_layout = info.GetAttrOrDefault<int64_t>("mrope_layout", 0);
  ORT_ENFORCE(info.GetAttrs<int64_t>("mrope_section", mrope_section).IsOK(),
              "MRotaryEmbedding: 'mrope_section' attribute is required");

  if (rotary_embedding_dim > 0) {
    ORT_ENFORCE(num_heads > 0, "num_heads must be provided if rotary_embedding_dim is specified");
  }
}

template <typename T>
Status MRotaryEmbedding<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* position_ids = context->Input<Tensor>(1);
  const Tensor* cos_cache = context->Input<Tensor>(2);
  const Tensor* sin_cache = context->Input<Tensor>(3);

  MRotaryParameters parameters = {};
  ORT_RETURN_IF_ERROR(CheckInputs<Tensor>(input,
                                          position_ids,
                                          cos_cache,
                                          sin_cache,
                                          num_heads,
                                          rotary_embedding_dim,
                                          mrope_section,
                                          mrope_layout,
                                          &parameters));

  Tensor* output = context->Output(0, input->Shape());
  if (input->Shape().Size() == 0) {
    return Status::OK();
  }

  if (is_packed_batching == false && parameters.sequence_length > parameters.max_sequence_length) {
    ORT_NOT_IMPLEMENTED("Updating cos_cache and sin_cache in MRotaryEmbedding is not currently supported");
  }

  typedef typename ToCudaType<T>::MappedType CudaT;
  auto& device_prop = GetDeviceProp();

  const int3 section = make_int3(parameters.mrope_section[0], parameters.mrope_section[1],
                                 parameters.mrope_section[2]);

  return LaunchMRotaryEmbeddingKernel<CudaT>(
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
      parameters.rotary_embedding_dim,
      parameters.max_sequence_length,
      interleaved,
      section,
      static_cast<int>(parameters.mrope_layout),
      scale,
      device_prop.maxThreadsPerBlock,
      parameters.transposed);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

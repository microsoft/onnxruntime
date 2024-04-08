// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/sparse/sparse_attention_impl.h"
#include "contrib_ops/cuda/sparse/sparse_attention.h"
#include "contrib_ops/cuda/sparse/sparse_attention_helper.h"

using namespace ::onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      SparseAttention,                                                   \
      kMSDomain,                                                         \
      1,                                                                 \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<int32_t>()}) \
          .MayInplace(3, 1)                                              \
          .MayInplace(4, 2),                                             \
      SparseAttention<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
// REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
SparseAttention<T>::SparseAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);

  int64_t sparse_block_size = 0;
  ORT_ENFORCE(info.GetAttr("sparse_block_size", &sparse_block_size).IsOK());
  ORT_ENFORCE(sparse_block_size == 16 || sparse_block_size == 32 || sparse_block_size == 64 || sparse_block_size == 128);
  sparse_block_size_ = static_cast<int>(sparse_block_size);

  is_causal_ = info.GetAttrOrDefault<int64_t>("causal", 1) == 1;

  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;

  softmax_scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
Status SparseAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* block_mask = context->Input<Tensor>(5);
  const Tensor* total_k_seq_lens = context->Input<Tensor>(6);
  const Tensor* cos_cache = context->Input<Tensor>(7);
  const Tensor* sin_cache = context->Input<Tensor>(8);

  auto& device_prop = GetDeviceProp();

  SparseAttentionParameters parameters;

  // Parameters from node attribute
  parameters.is_unidirectional = is_causal_;
  parameters.sparse_block_size = sparse_block_size_;
  parameters.num_heads = num_heads_;
  parameters.kv_num_heads = kv_num_heads_;
  parameters.scale = softmax_scale_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  ORT_RETURN_IF_ERROR(sparse_attention_helper::CheckInputs(&parameters,
                                                           query,
                                                           key,
                                                           value,
                                                           past_key,
                                                           past_value,
                                                           cos_cache,
                                                           sin_cache,
                                                           block_mask,
                                                           total_k_seq_lens));

  // Some limitations of CUDA kernels
  if (device_prop.maxThreadsPerBlock > 0 && num_heads_ > device_prop.maxThreadsPerBlock) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_heads should be no larger than ", device_prop.maxThreadsPerBlock);
  }

  // Compute output shape and get output tensors.
  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size);
  Tensor* output = context->Output(0, output_shape);

  assert(parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH);
  std::vector<int64_t> present_dims = {
      parameters.batch_size, parameters.kv_num_heads, parameters.max_sequence_length, parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(1, present_shape);
  Tensor* present_value = context->Output(2, present_shape);

  // Set input and output data.
  typedef typename ToCudaType<T>::MappedType CudaT;
  SparseAttentionData<CudaT> data;
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.past_key = (nullptr == past_key) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = (nullptr == past_value) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
  data.block_mask = block_mask->Data<int32_t>();
  data.seqlens_k_total = total_k_seq_lens->Data<int32_t>();
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present_key = (nullptr == present_key) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (nullptr == present_value) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());
  if (parameters.do_rotary) {
    data.cos_cache = reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
    data.sin_cache = reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  }

  parameters.past_present_share_buffer = (data.past_key != nullptr && data.past_key == data.present_key);
  if (parameters.past_present_share_buffer) {
    ORT_ENFORCE(data.past_value != nullptr && data.past_value == data.present_value);
  }

  cublasHandle_t cublas = GetCublasHandle(context);
  return QkvToContext<CudaT>(
      device_prop, cublas, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

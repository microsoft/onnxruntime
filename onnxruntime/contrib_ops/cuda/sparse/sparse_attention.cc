// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef USE_TRITON_KERNEL

#include "contrib_ops/cuda/sparse/sparse_attention_impl.h"
#include "contrib_ops/cuda/sparse/sparse_attention.h"
#include "contrib_ops/cuda/sparse/sparse_attention_helper.h"
#include "contrib_ops/cuda/sparse/block_mask.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                       \
      SparseAttention,                                                 \
      kMSDomain,                                                       \
      1,                                                               \
      T,                                                               \
      kCudaExecutionProvider,                                          \
      (*KernelDefBuilder::Create())                                    \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()) \
          .MayInplace(3, 1)                                            \
          .MayInplace(4, 2)                                            \
          .InputMemoryType(OrtMemTypeCPUInput, 6),                     \
      SparseAttention<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

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

  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;

  softmax_scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
}

template <typename T>
Status SparseAttention<T>::ComputeInternal(OpKernelContext* context) const {
  auto* tuning_ctx = GetTuningContext();
  if (!tuning_ctx->IsTunableOpEnabled()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparseAttention requires enabling tunable in provider option");
  }

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* block_mask = context->Input<Tensor>(5);
  const Tensor* total_seq_len = context->Input<Tensor>(6);
  const Tensor* seqlens_k_total = context->Input<Tensor>(7);
  const Tensor* cos_cache = context->Input<Tensor>(8);
  const Tensor* sin_cache = context->Input<Tensor>(9);

  auto& device_prop = GetDeviceProp();

  SparseAttentionParameters parameters;

  // Parameters from node attribute
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
                                                           seqlens_k_total,
                                                           total_seq_len));

  // Some limitations of CUDA kernels
  if (parameters.head_size != 128) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparseAttention in CUDA does not support the input: head_size=", parameters.head_size);
  }
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
  data.seqlens_k_total = (nullptr == seqlens_k_total) ? nullptr : seqlens_k_total->Data<int32_t>();
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present_key = (nullptr == present_key) ? nullptr : reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = (nullptr == present_value) ? nullptr : reinterpret_cast<CudaT*>(present_value->MutableData<T>());

  // Check past and present share buffer.
  parameters.past_present_share_buffer = (data.past_key != nullptr && data.past_key == data.present_key);
  if (parameters.past_present_share_buffer) {
    ORT_ENFORCE(data.past_value != nullptr && data.past_value == data.present_value);
  }
  if (!parameters.past_present_share_buffer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "CUDA implementation of SparseAttention requires past and present points to same buffer");
  }

  if (parameters.do_rotary) {
    data.cos_cache = reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
    data.sin_cache = reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  }

  // Currently, we use same block size in kernel.
  // TODO: support kernel block size that is smaller than sparse_block_size in tunable (need expand block mask).
  data.kernel_layout.block_size = parameters.sparse_block_size;
  data.kernel_layout.mask = data.block_mask;
  data.kernel_layout.num_layout = parameters.num_sparse_layout;
  data.kernel_layout.num_cols = parameters.max_sequence_length / data.kernel_layout.block_size;
  data.kernel_layout.num_rows = parameters.max_sequence_length / data.kernel_layout.block_size;

  // Allocate buffer for CSR col and row indices.
  onnxruntime::Stream* stream = context->GetComputeStream();
  int dense_blocks = data.kernel_layout.num_layout * data.kernel_layout.num_cols * data.kernel_layout.num_rows;
  auto csr_col_indices_buffer = GetScratchBuffer<int>(static_cast<size_t>(dense_blocks), stream);
  auto csr_row_indices_buffer = GetScratchBuffer<int>(
      static_cast<size_t>(data.kernel_layout.num_layout * (data.kernel_layout.num_rows + 1)), stream);

  data.kernel_layout.csr_col_indices = reinterpret_cast<const int*>(csr_col_indices_buffer.get());
  data.kernel_layout.csr_row_indices = reinterpret_cast<const int*>(csr_row_indices_buffer.get());

  int past_seq_len = parameters.total_sequence_length - parameters.sequence_length;
  if (past_seq_len > 0) {
    data.kernel_layout.start_row = (past_seq_len + 1) / data.kernel_layout.block_size;
  } else {
    data.kernel_layout.start_row = 0;
  }

  ConvertMaskToCSR(static_cast<cudaStream_t>(stream->GetHandle()),
                   data.kernel_layout.mask,
                   data.kernel_layout.num_layout,
                   data.kernel_layout.num_rows,
                   data.kernel_layout.num_cols,
                   csr_col_indices_buffer.get(),
                   csr_row_indices_buffer.get(),
                   device_prop.maxThreadsPerBlock);

  size_t rotary_buffer_bytes = 0;
  if (do_rotary_) {
    rotary_buffer_bytes = 2 * sizeof(T) * parameters.batch_size * parameters.num_heads * parameters.sequence_length * parameters.head_size;
    rotary_buffer_bytes += sizeof(int64_t) * parameters.batch_size * parameters.sequence_length;
  }
  auto rotary_buffer = GetScratchBuffer<void>(rotary_buffer_bytes, context->GetComputeStream());
  data.rotary_buffer = reinterpret_cast<CudaT*>(rotary_buffer.get());

  size_t transposed_q_bytes = 0;
  if (!parameters.is_packed_qkv) {
    transposed_q_bytes = (parameters.batch_size * parameters.sequence_length * parameters.num_heads * parameters.head_size * sizeof(T));
  }
  auto transposed_q_buffer = GetScratchBuffer<void>(transposed_q_bytes, context->GetComputeStream());
  if (transposed_q_buffer) {
    data.transposed_q_buffer = reinterpret_cast<CudaT*>(transposed_q_buffer.get());
  }

  size_t unpacked_qkv_bytes = 0;
  if (parameters.is_packed_qkv) {
    unpacked_qkv_bytes = (parameters.batch_size * parameters.sequence_length * (parameters.num_heads + 2 * parameters.kv_num_heads) * parameters.head_size * sizeof(T));
  }
  auto unpacked_qkv_buffer = GetScratchBuffer<void>(unpacked_qkv_bytes, context->GetComputeStream());
  if (unpacked_qkv_buffer) {
    data.unpacked_qkv_buffer = reinterpret_cast<CudaT*>(unpacked_qkv_buffer.get());
  }

  cublasHandle_t cublas = GetCublasHandle(context);
  return QkvToContext<CudaT>(
      device_prop, cublas, stream, parameters, data, tuning_ctx);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // USE_TRITON_KERNEL

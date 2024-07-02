// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/sparse/sparse_attention_impl.h"
#include "contrib_ops/cuda/sparse/sparse_attention.h"
#include "contrib_ops/cpu/sparse/sparse_attention_helper.h"
#include "contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_v1_api.h"
#include "contrib_ops/cuda/sparse/sparse_attention_v2/sparse_attention_v2_api.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/transformer_cuda_common.h"

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
          .InputMemoryType(OrtMemTypeCPUInput, 7),                     \
      SparseAttention<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

static inline int32_t DivUp(int32_t m, int32_t n) {
  return (m + n - 1) / n;
}

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
  ORT_ENFORCE(sparse_block_size == 64 || sparse_block_size == 128);
  sparse_block_size_ = static_cast<int>(sparse_block_size);

  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;

  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);

  disable_v1_kernel_ = ParseEnvironmentVariableWithDefault<bool>(sparse_attention::kDisableSparseAttentionV1, false);
}

template <typename T>
Status SparseAttention<T>::ComputeInternal(OpKernelContext* context) const {
  auto& device_prop = GetDeviceProp();
  if constexpr (std::is_same<T, BFloat16>::value) {
    if (device_prop.major < 8) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "bfloat16 requires Ampere and above GPUs with Compute Capability >= 8. Got major=",
                             device_prop.major);
    }
  }

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* block_row_indices = context->Input<Tensor>(5);
  const Tensor* block_col_indices = context->Input<Tensor>(6);
  const Tensor* total_seq_len = context->Input<Tensor>(7);
  const Tensor* seqlens_k_total = context->Input<Tensor>(8);
  const Tensor* cos_cache = context->Input<Tensor>(9);
  const Tensor* sin_cache = context->Input<Tensor>(10);

  SparseAttentionParameters parameters;

  // Parameters from node attribute shall be set before calling CheckInputs
  parameters.sparse_block_size = sparse_block_size_;
  parameters.num_heads = num_heads_;
  parameters.kv_num_heads = kv_num_heads_;
  parameters.scale = scale_;
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
                                                           block_row_indices,
                                                           block_col_indices,
                                                           seqlens_k_total,
                                                           total_seq_len));
  // Some limitations of CUDA kernels
  // The v1 and v2 kernels have same coverage, so only check one of them to see whether it is supported.
  int sm = device_prop.major * 10 + device_prop.minor;
  if (!sparse_attention_v1::is_supported_device(sm)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparseAttention only supports CUDA device with compute capacity 7.5, 8.0, 8.6, 8.9 and 9.0. Got sm=",
                           sm);
  }
  if (!sparse_attention_v1::is_supported_sparse_attention(parameters.head_size, sparse_block_size_)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SparseAttention only supports head_size=128 and sparse_block_size=64. Got head_size=",
                           parameters.head_size,
                           ",sparse_block_size=",
                           sparse_block_size_);
  }
  if (device_prop.maxThreadsPerBlock > 0 && num_heads_ > device_prop.maxThreadsPerBlock) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_heads should be no larger than ", device_prop.maxThreadsPerBlock);
  }

  int past_seq_len = parameters.total_sequence_length - parameters.sequence_length;
  bool is_prompt = (past_seq_len == 0);

  // The v1 kernel support only prompt and right padding only.
  // The v2 kernel support both prompt and token generation, and left/right padding.
  bool use_v2_kernel = disable_v1_kernel_ || !is_prompt;

  // Async Copy total_k_seq_len from GPU to CPU.
  IAllocatorUniquePtr<int32_t> pinned_buffer;
  int32_t* total_k_seq_len_pinned = nullptr;
  AutoDestoryCudaEvent new_event;
  cudaEvent_t& isCopyDone = new_event.Get();
  cudaStream_t cuda_stream = Stream(context);
  if (use_v2_kernel) {
    pinned_buffer = AllocateBufferOnCPUPinned<int32_t>(parameters.batch_size);

    total_k_seq_len_pinned = reinterpret_cast<int32_t*>(pinned_buffer.get());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(total_k_seq_len_pinned,
                                         seqlens_k_total->Data<int32_t>(),
                                         sizeof(int32_t) * parameters.batch_size,
                                         cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaEventCreate(&isCopyDone));
    CUDA_RETURN_IF_ERROR(cudaEventRecord(isCopyDone, cuda_stream));
  }

  if constexpr (std::is_same<T, MLFloat16>::value) {
    // std::call_once is used in load_sparse_attention_fp16 so no need to use mutex here.
    // After kernel is loaded, it will stay in memory until the process exits. We do not unload explicitly.
    // TODO(tianleiwu): use TSharedCubinKernelFactory to manage kernel loading/unloading.
    if (use_v2_kernel) {
      sparse_attention_v2::load_sparse_attention_fp16(sm);
    } else {
      sparse_attention_v1::load_sparse_attention_fp16(sm);
    }
  } else {
    if (use_v2_kernel) {
      sparse_attention_v2::load_sparse_attention_bf16(sm);
    } else {
      sparse_attention_v1::load_sparse_attention_bf16(sm);
    }
  }

  // Compute output shape and get output tensors.
  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size);
  Tensor* output = context->Output(0, output_shape);

  std::vector<int64_t> present_dims = {
      parameters.batch_size, parameters.kv_num_heads, parameters.max_cache_sequence_length, parameters.head_size};
  TensorShape present_shape(present_dims);
  Tensor* present_key = context->Output(1, present_shape);
  Tensor* present_value = context->Output(2, present_shape);

  // Set input and output data.
  typedef typename ToCudaType<T>::MappedType CudaT;
  SparseAttentionData<CudaT> data;
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.past_key = reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.past_value = reinterpret_cast<const CudaT*>(past_value->Data<T>());
  data.seqlens_k_total = (nullptr == seqlens_k_total) ? nullptr : seqlens_k_total->Data<int32_t>();
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.present_key = reinterpret_cast<CudaT*>(present_key->MutableData<T>());
  data.present_value = reinterpret_cast<CudaT*>(present_value->MutableData<T>());

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
  data.kernel_layout.num_layout = parameters.num_sparse_layout;
  data.kernel_layout.csr_col_indices = block_col_indices->Data<int32_t>();
  data.kernel_layout.csr_row_indices = block_row_indices->Data<int32_t>();

  size_t rotary_buffer_bytes = 0;
  if (do_rotary_) {
    rotary_buffer_bytes = 2 * sizeof(T) * parameters.batch_size * parameters.num_heads *
                          parameters.sequence_length * parameters.head_size;
    rotary_buffer_bytes += sizeof(int64_t) * parameters.batch_size * parameters.sequence_length;
  }
  onnxruntime::Stream* stream = context->GetComputeStream();
  auto rotary_buffer = GetScratchBuffer<void>(rotary_buffer_bytes, stream);
  data.rotary_buffer = reinterpret_cast<CudaT*>(rotary_buffer.get());

  size_t transposed_q_bytes = 0;
  if (!parameters.is_packed_qkv) {
    transposed_q_bytes = parameters.batch_size * parameters.sequence_length *
                         parameters.num_heads * parameters.head_size * sizeof(T);
  }
  auto transposed_q_buffer = GetScratchBuffer<void>(transposed_q_bytes, stream);
  if (transposed_q_buffer) {
    data.transposed_q_buffer = reinterpret_cast<CudaT*>(transposed_q_buffer.get());
  }

  size_t unpacked_qkv_bytes = 0;
  if (parameters.is_packed_qkv) {
    unpacked_qkv_bytes = (parameters.batch_size * parameters.sequence_length *
                          (parameters.num_heads + 2 * parameters.kv_num_heads) *
                          parameters.head_size * sizeof(T));
  }
  auto unpacked_qkv_buffer = GetScratchBuffer<void>(unpacked_qkv_bytes, stream);
  if (unpacked_qkv_buffer) {
    data.unpacked_qkv_buffer = reinterpret_cast<CudaT*>(unpacked_qkv_buffer.get());
  }

  // Prepare some v2 kernel inputs in CPU then copy to GPU.
  IAllocatorUniquePtr<int32_t> v2_kernel_inputs_pinned_buffer;
  IAllocatorUniquePtr<int32_t> v2_kernel_buffer;
  data.use_v2_kernel = use_v2_kernel;
  if (use_v2_kernel) {
    // Compute activate q blocks so that we know the size of buffer to allocate.
    CUDA_RETURN_IF_ERROR(cudaEventSynchronize(isCopyDone));
    int active_q_blocks = 0;
    if (is_prompt) {
      for (int i = 0; i < parameters.batch_size; i++) {
        active_q_blocks += DivUp(is_prompt ? total_k_seq_len_pinned[i] : 1, data.kernel_layout.block_size);
      }
    } else {  // not prompt
      assert(parameters.sequence_length == 1);
      active_q_blocks = parameters.batch_size;
    }

    // Compute buffer size: addresses of 6 buffers for v2 kernel need to be aligned to 16.
    const size_t aligned_batch_size = DivUp(parameters.batch_size, 16) * 16;
    const size_t aligned_num_q_blocks = DivUp(active_q_blocks, 16) * 16;
    size_t v2_kernel_buffer_size = 4 * aligned_batch_size + 2 * aligned_num_q_blocks;

    // Compute those values in CPU, then copy to GPU
    v2_kernel_inputs_pinned_buffer = AllocateBufferOnCPUPinned<int32_t>(v2_kernel_buffer_size);
    int32_t* v2_kernel_inputs_pinned = reinterpret_cast<int32_t*>(v2_kernel_inputs_pinned_buffer.get());
    int32_t* q_batch_starts = v2_kernel_inputs_pinned;
    int32_t* q_batch_ends = q_batch_starts + aligned_batch_size;
    int32_t* k_batch_starts = q_batch_ends + aligned_batch_size;
    int32_t* k_batch_ends = k_batch_starts + aligned_batch_size;
    int32_t* q_batch_ids = k_batch_ends + aligned_batch_size;
    int32_t* q_start_sids = q_batch_ids + aligned_num_q_blocks;

    // Here assumes right-side padding
    if (is_prompt) {
      for (int i = 0; i < parameters.batch_size; i++) {
        q_batch_starts[i] = 0;
        q_batch_ends[i] = total_k_seq_len_pinned[i];
        k_batch_starts[i] = 0;
        k_batch_ends[i] = total_k_seq_len_pinned[i];
      }
    } else {
      for (int i = 0; i < parameters.batch_size; i++) {
        q_batch_starts[i] = 0;
        q_batch_ends[i] = 1;
        k_batch_starts[i] = 0;
        k_batch_ends[i] = total_k_seq_len_pinned[i];
      }
    }

    int current_block = 0;
    for (int i = 0; i < parameters.batch_size; i++) {
      int blocks = DivUp(q_batch_ends[i] - q_batch_starts[i], data.kernel_layout.block_size);
      for (int j = 0; j < blocks; j++) {
        q_batch_ids[current_block] = i;
        q_start_sids[current_block] = j * data.kernel_layout.block_size;
        current_block++;
      }
    }

    v2_kernel_buffer = GetScratchBuffer<int>(v2_kernel_buffer_size, stream);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(v2_kernel_buffer.get(), v2_kernel_inputs_pinned,
                                         sizeof(int32_t) * v2_kernel_buffer_size,
                                         cudaMemcpyHostToDevice, cuda_stream));

    data.q_batch_starts = v2_kernel_buffer.get();
    data.q_batch_ends = data.q_batch_starts + aligned_batch_size;
    data.k_batch_starts = data.q_batch_ends + aligned_batch_size;
    data.k_batch_ends = data.k_batch_starts + aligned_batch_size;
    data.q_batch_ids = data.k_batch_ends + aligned_batch_size;
    data.q_start_sids = data.q_batch_ids + aligned_num_q_blocks;
    data.active_q_blocks = active_q_blocks;
  }

  return QkvToContext<CudaT>(device_prop, stream, parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

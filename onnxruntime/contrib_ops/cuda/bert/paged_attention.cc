// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cpu/utils/dump_tensor.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "contrib_ops/cuda/bert/paged_attention.h"
#include "contrib_ops/cuda/bert/paged_attention_helper.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      PagedAttention,                                                   \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("S", DataTypeImpl::GetTensorType<int32_t>()), \
      PagedAttention<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
PagedAttention<T>::PagedAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);

  kernel_options_ = this->GetAttentionKernelOptions();
  disable_flash_attention_ = sizeof(T) != 2 || !kernel_options_->UseFlashAttention();
  disable_memory_efficient_attention_ = sizeof(T) != 2 || !kernel_options_->UseEfficientAttention();
}

template <typename T>
Status PagedAttention<T>::ComputeInternal(OpKernelContext* context) const {
  auto ort_stream = GetOrtStream(context);

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* key_cache = context->Input<Tensor>(3);
  const Tensor* value_cache = context->Input<Tensor>(4);
  const Tensor* cumulative_seqlens_q = context->Input<Tensor>(5);
  const Tensor* past_seqlens = context->Input<Tensor>(6);
  const Tensor* block_table = context->Input<Tensor>(7);
  const Tensor* cos_cache = context->Input<Tensor>(8);
  const Tensor* sin_cache = context->Input<Tensor>(9);

  auto& device_prop = GetDeviceProp();
  PagedAttentionParameters parameters;
  typedef typename ToCudaType<T>::MappedType CudaT;
  PagedAttentionData<CudaT> data;

  // Check shapes of inputs to op and set parameters
  ORT_RETURN_IF_ERROR(paged_attention_helper::CheckInputs(query,
                                                          key,
                                                          value,
                                                          key_cache,
                                                          value_cache,
                                                          cumulative_seqlens_q,
                                                          past_seqlens,
                                                          block_table,
                                                          cos_cache,
                                                          sin_cache,
                                                          &parameters,
                                                          num_heads_,
                                                          kv_num_heads_,
                                                          scale_,
                                                          softcap_,
                                                          device_prop.maxThreadsPerBlock));
  parameters.local_window_size = local_window_size_;
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  DUMP_STRING_INIT();
  DUMP_STRING("Batch size = ", parameters.batch_size);
  DUMP_STRING("Token count = ", parameters.token_count);
  DUMP_STRING("Q hidden size = ", parameters.hidden_size);
  DUMP_STRING("KV hidden size = ", parameters.kv_hidden_size);
  DUMP_STRING("Q num heads = ", parameters.num_heads);
  DUMP_STRING("KV num heads = ", parameters.kv_num_heads);
  DUMP_STRING("Head size = ", parameters.head_size);
  DUMP_STRING("Num blocks = ", parameters.num_blocks);
  DUMP_STRING("Block size = ", parameters.block_size);
  DUMP_STRING("Max num blocks per sequence = ", parameters.max_num_blocks_per_seq);
  DUMP_STRING("Rotary dimension = ", parameters.rotary_dim);
  DUMP_STRING("Is packed QKV = ", parameters.is_packed_qkv);

  // Check rotary
  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache must be passed to PagedAttention when do_rotary = 1");
  }

  // Set output tensor shapes
  TensorShapeVector output_shape(2);
  output_shape[0] = static_cast<int64_t>(parameters.token_count);
  output_shape[1] = static_cast<int64_t>(parameters.hidden_size);
  Tensor* output = context->Output(0, output_shape);

  TensorShapeVector key_cache_out_shape(4);
  key_cache_out_shape[0] = static_cast<int64_t>(parameters.num_blocks);
  key_cache_out_shape[1] = static_cast<int64_t>(parameters.block_size);
  key_cache_out_shape[2] = static_cast<int64_t>(parameters.kv_num_heads);
  key_cache_out_shape[3] = static_cast<int64_t>(parameters.head_size);
  Tensor* key_cache_out = context->Output(1, key_cache_out_shape);

  TensorShapeVector value_cache_out_shape(4);
  value_cache_out_shape[0] = static_cast<int64_t>(parameters.num_blocks);
  value_cache_out_shape[1] = static_cast<int64_t>(parameters.block_size);
  value_cache_out_shape[2] = static_cast<int64_t>(parameters.kv_num_heads);
  value_cache_out_shape[3] = static_cast<int64_t>(parameters.head_size);
  Tensor* value_cache_out = context->Output(2, value_cache_out_shape);

  if (key_cache_out != nullptr && key_cache->Data<T>() != key_cache_out->MutableData<T>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "key_cache and key_cache_out must be the same buffer");
  } else if (value_cache_out != nullptr && value_cache->Data<T>() != value_cache_out->MutableData<T>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "value_cache and value_cache_out must be the same buffer");
  }

  // Empty query input: output is already shaped [0, hidden_size], and the cache outputs
  // alias the input caches (verified above), so no backend kernel or cache update is needed.
  if (parameters.token_count == 0) {
    return Status::OK();
  }

  // Kernel backend selection — FlashAttention preferred, fall back to MemoryEfficientAttention.
#if USE_FLASH_ATTENTION
  bool use_flash_attention = !disable_flash_attention_ &&
                             onnxruntime::flash::is_supported<T>(device_prop,
                                                                 parameters.head_size,
                                                                 parameters.num_heads,
                                                                 parameters.kv_num_heads);
#else
  constexpr bool use_flash_attention = false;
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  const int sm = device_prop.major * 10 + device_prop.minor;
  const bool is_half = std::is_same<T, MLFloat16>::value;
  const bool is_bf16 = std::is_same<T, BFloat16>::value;
  bool use_memory_efficient_attention =
      !use_flash_attention &&
      !disable_memory_efficient_attention_ &&
      has_memory_efficient_attention(sm, is_half, is_bf16,
                                     parameters.head_size, parameters.head_size);
#else
  constexpr bool use_memory_efficient_attention = false;
#endif

  if (!use_flash_attention && !use_memory_efficient_attention) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "PagedAttention requires FlashAttention (sm>=80, fp16/bf16) or "
                           "MemoryEfficientAttention (fp16 sm>=53, bf16 sm>=80, head_size<=1024 and %8==0) "
                           "to be available. Check ORT_DISABLE_FLASH_ATTENTION / "
                           "ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION env vars and dtype/head_size.");
  }

  // Scratch buffers common to both backends.
  size_t softmax_lse_bytes = 0;
#if USE_FLASH_ATTENTION
  if (use_flash_attention) {
    softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(parameters.token_count,
                                                                 parameters.num_heads);
  }
#endif
  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, GetComputeStream(context));

  size_t cumulative_seqlens_kv_bytes = sizeof(int) * (parameters.batch_size + 1);
  auto cumulative_seqlens_kv_buffer = GetScratchBuffer<void>(cumulative_seqlens_kv_bytes, GetComputeStream(context));
  int* cumulative_seqlens_kv_ptr = reinterpret_cast<int*>(cumulative_seqlens_kv_buffer.get());

  size_t workspace_buffer_bytes = 0;
  if (do_rotary_) {
    workspace_buffer_bytes = sizeof(T) * parameters.token_count * (parameters.hidden_size + parameters.kv_hidden_size);
  } else if (parameters.is_packed_qkv) {
    workspace_buffer_bytes = sizeof(T) * parameters.token_count * parameters.hidden_size;
  }
  auto workspace_buffer = GetScratchBuffer<void>(workspace_buffer_bytes, GetComputeStream(context));

  // Populate cumulative_seqlens_kv for both backends. The MEA path additionally needs
  // the last element on the host to size the tight gather buffers, so we D->H sync below.
  //
  // LaunchGetCumulativeSeqlensKV uses a per-block cub::BlockScan with a block size of 256
  // and launches (batch_size + 255) / 256 blocks, so blocks scan independently. Enforce
  // batch_size <= 256 so the cumulative sum is correct; a larger batch would silently
  // produce wrong KV offsets. (A future grid-wide scan could lift this limit.)
  constexpr int kMaxBatchSizeForCumulativeSeqlensKV = 256;
  if (parameters.batch_size > kMaxBatchSizeForCumulativeSeqlensKV) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "PagedAttention currently supports batch_size <= ",
                           kMaxBatchSizeForCumulativeSeqlensKV,
                           " (LaunchGetCumulativeSeqlensKV limitation); got batch_size=",
                           parameters.batch_size, ".");
  }

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(ort_stream.get()->GetHandle());
  ORT_RETURN_IF_ERROR(LaunchGetCumulativeSeqlensKV(
      cumulative_seqlens_kv_ptr,
      reinterpret_cast<const int*>(cumulative_seqlens_q->Data<int>()),
      reinterpret_cast<const int*>(past_seqlens->Data<int>()),
      parameters.batch_size, cuda_stream));

  int total_kv_tokens = 0;
  int max_query_len = 0;
  IAllocatorUniquePtr<void> gathered_key_buffer;
  IAllocatorUniquePtr<void> gathered_value_buffer;
  IAllocatorUniquePtr<void> fmha_buffer;

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (use_memory_efficient_attention) {
    // MEA needs two host-side quantities:
    //   - total_kv_tokens (= cumulative_seqlens_kv[batch_size]) to size tight gather buffers.
    //   - max_query_len (= max per-batch new-query length) to size the rotary and MEA grids
    //     correctly. The heuristic `token_count - batch_size + 1` underestimates when any
    //     batch has 0 new tokens (valid input), silently dropping query-tokens from those
    //     larger-than-average batches.
    // Both come from cumulative_seqlens_q / cumulative_seqlens_kv, which are tiny (batch+1
    // ints each), so one D->H copy of the full arrays is cheaper than issuing an extra
    // reduction kernel and avoids a second sync.
    const int kCumulativeCount = parameters.batch_size + 1;
    auto cum_q_pinned = this->AllocateBufferOnCPUPinned<int>(kCumulativeCount);
    auto cum_kv_pinned = this->AllocateBufferOnCPUPinned<int>(kCumulativeCount);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cum_q_pinned.get(),
                                         reinterpret_cast<const int*>(cumulative_seqlens_q->Data<int>()),
                                         sizeof(int) * kCumulativeCount, cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(cum_kv_pinned.get(), cumulative_seqlens_kv_ptr,
                                         sizeof(int) * kCumulativeCount, cudaMemcpyDeviceToHost, cuda_stream));
    CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(cuda_stream));
    total_kv_tokens = cum_kv_pinned.get()[parameters.batch_size];
    for (int i = 0; i < parameters.batch_size; ++i) {
      const int q_len_i = cum_q_pinned.get()[i + 1] - cum_q_pinned.get()[i];
      if (q_len_i > max_query_len) {
        max_query_len = q_len_i;
      }
    }
    if (total_kv_tokens == 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "PagedAttention MEA fallback: total_kv_tokens is zero for non-empty input.");
    }
    if (total_kv_tokens < 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "PagedAttention MEA fallback: total_kv_tokens is negative (", total_kv_tokens, ").");
    }

    const size_t gather_elems = static_cast<size_t>(total_kv_tokens) *
                                parameters.num_heads * parameters.head_size;
    gathered_key_buffer = GetScratchBuffer<void>(sizeof(T) * gather_elems, GetComputeStream(context));
    gathered_value_buffer = GetScratchBuffer<void>(sizeof(T) * gather_elems, GetComputeStream(context));

    if (MemoryEfficientAttentionParams::need_workspace(parameters.head_size, sizeof(T) == sizeof(float))) {
      // MEA output accumulator is float32 regardless of input dtype (see GQA pattern at
      // group_query_attention.cc:482); use sizeof(float), not sizeof(T).
      const size_t fmha_elems = static_cast<size_t>(parameters.token_count) *
                                parameters.num_heads * parameters.head_size;
      fmha_buffer = GetScratchBuffer<void>(sizeof(float) * fmha_elems, GetComputeStream(context));
    }
  }
#endif

  // Print debug info
  if (kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = use_flash_attention;
    debug_info.use_efficient_attention = use_memory_efficient_attention;

    debug_info.Print("PagedAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  // Set up data struct for kernel launch
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());
  data.key_cache = reinterpret_cast<CudaT*>(const_cast<T*>(key_cache->Data<T>()));
  data.value_cache = reinterpret_cast<CudaT*>(const_cast<T*>(value_cache->Data<T>()));
  data.cumulative_seqlens_q = reinterpret_cast<const int*>(cumulative_seqlens_q->Data<int>());
  data.past_seqlens = reinterpret_cast<const int*>(past_seqlens->Data<int>());
  data.cumulative_seqlens_kv = cumulative_seqlens_kv_ptr;
  data.block_table = reinterpret_cast<const int*>(block_table->Data<int>());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.use_flash_attention = use_flash_attention;
  data.use_memory_efficient_attention = use_memory_efficient_attention;
  if (softmax_lse_buffer != nullptr) {
    data.softmax_lse = reinterpret_cast<CudaT*>(softmax_lse_buffer.get());
  }
  if (workspace_buffer != nullptr) {
    data.workspace_buffer = reinterpret_cast<CudaT*>(workspace_buffer.get());
  }
  if (parameters.do_rotary) {
    data.cos_cache = reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
    data.sin_cache = reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  }
  if (use_memory_efficient_attention) {
    data.gathered_key = reinterpret_cast<CudaT*>(gathered_key_buffer.get());
    data.gathered_value = reinterpret_cast<CudaT*>(gathered_value_buffer.get());
    if (fmha_buffer != nullptr) {
      data.fmha_buffer = reinterpret_cast<CudaT*>(fmha_buffer.get());
    }
    data.total_kv_tokens = total_kv_tokens;
    data.max_query_len = max_query_len;
  }

  cublasHandle_t cublas = GetCublasHandle(context);

  return QkvToContext<CudaT>(
      device_prop, cublas, ort_stream.get(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

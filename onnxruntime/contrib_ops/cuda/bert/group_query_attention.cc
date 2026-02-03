// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <algorithm>
#include "core/providers/cuda/cuda_common.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cpu/utils/debug_macros.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      GroupQueryAttention,                                               \
      kMSDomain,                                                         \
      1,                                                                 \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("M", {DataTypeImpl::GetTensorType<int32_t>()}) \
          .MayInplace(3, 1)                                              \
          .MayInplace(4, 2)                                              \
          .InputMemoryType(OrtMemTypeCPUInput, 6),                       \
      GroupQueryAttention<T>);

REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

constexpr const char* kDisableFlashDecode = "ORT_DISABLE_FLASH_DECODE";

// Group Query Attention (GQA) Operator
//
// This operator implements Group Query Attention, a variation of Multi-Head Attention (MHA)
// where the number of key/value heads is smaller than the number of query heads.
// It supports:
// - Rotary Positional Embeddings (RoPE)
// - KV Cache (past/present key/value)
// - Quantized KV Cache (Int8/Int4) via GroupQueryAttentionData
// - Flash Attention and Memory Efficient Attention backends
//
template <typename T>
GroupQueryAttention<T>::GroupQueryAttention(const OpKernelInfo& info)
    : CudaKernel(info) {
  int64_t num_heads = 0;
  int64_t kv_num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  ORT_ENFORCE(info.GetAttr("kv_num_heads", &kv_num_heads).IsOK() && kv_num_heads > 0 && num_heads % kv_num_heads == 0);
  num_heads_ = static_cast<int>(num_heads);
  kv_num_heads_ = static_cast<int>(kv_num_heads);
  is_past_bsnh_ = false;
  is_unidirectional_ = true;
  local_window_size_ = static_cast<int>(info.GetAttrOrDefault<int64_t>("local_window_size", -1));
  do_rotary_ = info.GetAttrOrDefault<int64_t>("do_rotary", 0) == 1;
  rotary_interleaved_ = info.GetAttrOrDefault<int64_t>("rotary_interleaved", 0) == 1;
  scale_ = info.GetAttrOrDefault<float>("scale", 0.0f);
  softcap_ = info.GetAttrOrDefault<float>("softcap", 0.0f);
  use_smooth_softmax_ = info.GetAttrOrDefault<int64_t>("smooth_softmax", 0) == 1;

  kernel_options_ = this->GetAttentionKernelOptions();

  disable_flash_attention_ = sizeof(T) != 2 || !kernel_options_->UseFlashAttention();

  // Memory efficient attention supports float and float16. BFloat16 support added for SM80+.
  disable_memory_efficient_attention_ = !kernel_options_->UseEfficientAttention();

  if (!disable_flash_attention_) {
    zeros_ = this->GetScratchBuffer<int>(kZerosCount, nullptr);
    CUDA_CALL_THROW(cudaMemset(zeros_.get(), 0, kZerosCount * sizeof(int)));
  }

  disable_flash_decode_ = ParseEnvironmentVariableWithDefault<bool>(kDisableFlashDecode, false);
}

// ComputeInternal executes the GQA kernel.
//
// Inputs:
// 0. query             (Tensor) [batch, sequence_length, hidden_size]
// 1. key               (Tensor) [batch, sequence_length, kv_hidden_size] (Optional)
// 2. value             (Tensor) [batch, sequence_length, kv_hidden_size] (Optional)
// 3. past_key          (Tensor) [batch, num_kv_heads, max_seq_len, head_size] (Optional)
// 4. past_value        (Tensor) [batch, num_kv_heads, max_seq_len, head_size] (Optional)
// 5. seqlens_k         (Tensor) [batch] - Total sequence length minus 1 (for historical compatibility)
// 6. total_seqlen      (Tensor) - Max total sequence length
// 7. cos_cache         (Tensor) - Precomputed cosine table for RoPE
// 8. sin_cache         (Tensor) - Precomputed sine table for RoPE
// 9. position_ids      (Tensor) - Position indices for RoPE
// 10. attention_bias   (Tensor) - Not supported in this kernel
// 11. head_sink        (Tensor) - Attention sink for GPT-OSS
template <typename T>
Status GroupQueryAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);

  // The input seqlens_k is total sequence length - 1 for historical reasons.
  // Rename it to total_seq_lens_minus_one in cuda kernel to avoid confusion.
  const Tensor* total_seq_lens_minus_one = context->Input<Tensor>(5);

  // The max of total sequence lengths. The content of this tensor is a scalar stored in CPU memory.
  const Tensor* total_seqlen = context->Input<Tensor>(6);

  const Tensor* cos_cache = context->Input<Tensor>(7);
  const Tensor* sin_cache = context->Input<Tensor>(8);
  const Tensor* position_ids = context->Input<Tensor>(9);
  const Tensor* attention_bias = context->Input<Tensor>(10);
  const Tensor* head_sink = context->Input<Tensor>(11);

  if (attention_bias != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "attention_bias is not supported in GroupQueryAttention cuda kernel.");
  }

  auto& device_prop = GetDeviceProp();
  GroupQueryAttentionParameters parameters;
  typedef typename ToCudaType<T>::MappedType CudaT;
  GroupQueryAttentionData<CudaT> data;

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                cos_cache,
                                                                sin_cache,
                                                                &parameters,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                total_seq_lens_minus_one,
                                                                total_seqlen,
                                                                scale_,
                                                                softcap_,
                                                                device_prop.maxThreadsPerBlock));

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckCustomAttentionInputs(position_ids,
                                                                               attention_bias,
                                                                               head_sink,
                                                                               parameters));
  parameters.local_window_size = local_window_size_;
  parameters.is_unidirectional = is_unidirectional_;
  parameters.use_smooth_softmax = use_smooth_softmax_ || head_sink != nullptr;
  parameters.zeros_count = kZerosCount;
  parameters.zero_ptr = zeros_.get();
  parameters.do_rotary = do_rotary_;
  parameters.rotary_interleaved = rotary_interleaved_;

  // The current GQA CUDA implementation will never be able to have a QK output.
  // GQA CUDA uses either flash attention or memory efficient attention. Neither kernel supports returning the QK output.
  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckNoQKOutput(
      context->OutputCount(),
      static_cast<int>(Info().GetAttrOrDefault<int64_t>("qk_output", static_cast<int64_t>(QKOutputType::NO_OUTPUT)))));

  if (do_rotary_ && (cos_cache == nullptr || sin_cache == nullptr)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "cos_cache and sin_cache must be passed to GroupQueryAttention when do_rotary = 1");
  }

  TensorShapeVector output_shape(3);
  output_shape[0] = static_cast<int64_t>(parameters.batch_size);
  output_shape[1] = static_cast<int64_t>(parameters.sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.hidden_size);
  Tensor* output = context->Output(0, output_shape);

  // Set up present KV output shapes
  std::vector<int64_t> present_dims = {
      parameters.batch_size, parameters.kv_num_heads, parameters.seqlen_present_kv_cache, parameters.head_size};

  TensorShape present_shape(present_dims);
  context->Output(1, present_shape);  // present_key
  context->Output(2, present_shape);  // present_value

  IAllocatorUniquePtr<void> k_buffer;
  IAllocatorUniquePtr<void> v_buffer;
  IAllocatorUniquePtr<void> rotary_buffer;
  IAllocatorUniquePtr<void> fmha_buffer;
  IAllocatorUniquePtr<void> unpacked_qkv_buffer;
  IAllocatorUniquePtr<int> seq_lens_buffer;

  // Flash Attention buffers
  IAllocatorUniquePtr<void> softmax_lse_buffer;
  IAllocatorUniquePtr<void> softmax_lse_accum_buffer;
  IAllocatorUniquePtr<void> out_accum_buffer;

  data.position_ids = (position_ids != nullptr) ? position_ids->Data<int64_t>() : nullptr;

  // Input pointers for both paths
  data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
  data.key = key == nullptr ? nullptr : reinterpret_cast<const CudaT*>(key->Data<T>());
  data.value = value == nullptr ? nullptr : reinterpret_cast<const CudaT*>(value->Data<T>());

  // Handle Past/Present pointers
  data.past_key = (past_key == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_key->Data<T>());
  data.present_key = reinterpret_cast<CudaT*>(context->Output<Tensor>(1)->MutableData<T>());
  data.past_value = (past_value == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(past_value->Data<T>());
  data.present_value = reinterpret_cast<CudaT*>(context->Output<Tensor>(2)->MutableData<T>());

  // Compute past_present_share_buffer early since it's needed for flash attention path selection.
  // This compares the final pointer values after quantization handling.
  parameters.past_present_share_buffer = (data.past_key == data.present_key);

#if USE_FLASH_ATTENTION
  bool use_flash_attention = !disable_flash_attention_ &&
                             onnxruntime::flash::is_supported<T>(device_prop,
                                                                 parameters.head_size,
                                                                 parameters.num_heads,
                                                                 parameters.kv_num_heads);

  data.use_flash_attention = use_flash_attention;
  data.use_flash_attention_fast_decode = use_flash_attention && !disable_flash_decode_ && !parameters.is_first_prompt && parameters.past_present_share_buffer;

  if (use_flash_attention) {
    // Allocate Flash specific buffers (Softmax LSE, Accum)
    size_t softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(parameters.sequence_length, parameters.batch_size, parameters.num_heads);

    int num_heads_for_split = data.use_flash_attention_fast_decode ? parameters.kv_num_heads : parameters.num_heads;
    auto [num_splits, softmax_lse_accum_bytes, out_accum_bytes] = onnxruntime::flash::get_num_splits_and_buffer_sizes(
        parameters.batch_size, parameters.sequence_length, parameters.total_sequence_length, num_heads_for_split,
        parameters.head_size, device_prop.multiProcessorCount);

    parameters.num_splits = static_cast<int>(num_splits);

    if (data.use_flash_attention_fast_decode && num_splits > 1) {
      // The heuristic used kv_num_heads to maximize occupancy for the GQA-aware kernel.
      // However, the LSE and Accum buffers must store results for ALL num_heads.
      softmax_lse_accum_bytes = onnxruntime::flash::get_softmax_lse_accum_size(num_splits, parameters.batch_size, parameters.num_heads, parameters.sequence_length);
      auto round_multiple = [](size_t x, size_t m) { return (x + m - 1) / m * m; };
      out_accum_bytes = onnxruntime::flash::get_out_accum_size(num_splits, parameters.batch_size, parameters.num_heads, parameters.sequence_length, round_multiple(parameters.head_size, 32));
    }

    softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
    softmax_lse_accum_buffer = GetScratchBuffer<void>(softmax_lse_accum_bytes, context->GetComputeStream());
    out_accum_buffer = GetScratchBuffer<void>(out_accum_bytes, context->GetComputeStream());

    data.softmax_lse = reinterpret_cast<CudaT*>(softmax_lse_buffer.get());
    data.softmax_lse_accum = reinterpret_cast<CudaT*>(softmax_lse_accum_buffer.get());
    data.out_accum = reinterpret_cast<CudaT*>(out_accum_buffer.get());
  }
#endif

  if (data.use_flash_attention_fast_decode && parameters.sequence_length == 1) {
    // FlashDecoding Fast Path:
    // - Uses Flash Attention's internal KV append logic, so total_seq_lens and padded_seq_lens are not needed.
    // - The input seqlens_k from ONNX graph is (total_len - 1), which equals past_seq_len when seq_len == 1.
    // - This optimization avoids launching GetSequenceLengths kernel for single-token decoding.
    data.past_seq_lens = const_cast<int*>(total_seq_lens_minus_one->Data<int>());
  } else {
    // Compute sequence length buffers (past_seq_lens and total_seq_lens).
    // Allocate buffer for both: first half is past_seq_lens, second half is total_seq_lens.
    seq_lens_buffer = GetScratchBuffer<int>(3 * parameters.batch_size, context->GetComputeStream());
    auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());
    data.past_seq_lens = seq_lens_buffer.get();
    data.total_seq_lens = seq_lens_buffer.get() + parameters.batch_size;
    data.padded_seq_lens = data.total_seq_lens + parameters.batch_size;
    ORT_RETURN_IF_ERROR(LaunchGetSequenceLengths(total_seq_lens_minus_one->Data<int>(),
                                                 data.past_seq_lens,
                                                 data.total_seq_lens,
                                                 data.padded_seq_lens,
                                                 parameters.batch_size,
                                                 parameters.sequence_length,
                                                 parameters.is_first_prompt,
                                                 cuda_stream,
                                                 device_prop.maxThreadsPerBlock));
    DUMP_TENSOR_INIT();
    DUMP_TENSOR("total_seq_lens", data.total_seq_lens, parameters.batch_size, 1);
    DUMP_TENSOR("past_seq_lens", data.past_seq_lens, parameters.batch_size, 1);
    DUMP_TENSOR("padded_seq_lens", data.padded_seq_lens, parameters.batch_size, 1);
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (!data.use_flash_attention) {
    // Fall back to memory efficient attention.
    int sm = (device_prop.major * 10) + device_prop.minor;
    bool use_memory_efficient_attention =
        !disable_memory_efficient_attention_ &&
        has_memory_efficient_attention(sm, std::is_same<T, MLFloat16>::value, std::is_same<T, BFloat16>::value, parameters.head_size, parameters.head_size);
    data.use_memory_efficient_attention = use_memory_efficient_attention;

    // KV buffer for head expansion (when num_heads != kv_num_heads)
    size_t kv_buffer_bytes = (use_memory_efficient_attention && (parameters.num_heads != parameters.kv_num_heads))
                                 ? (sizeof(T) * parameters.batch_size * parameters.num_heads * parameters.seqlen_present_kv_cache * parameters.head_size)
                                 : 0;
    // FMHA workspace
    size_t fmha_buffer_bytes = (use_memory_efficient_attention && MemoryEfficientAttentionParams::need_workspace(parameters.head_size, sizeof(T) == sizeof(float)))
                                   ? (sizeof(float) * parameters.batch_size * parameters.sequence_length * parameters.num_heads * parameters.head_size)
                                   : 0;

    k_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
    v_buffer = GetScratchBuffer<void>(kv_buffer_bytes, context->GetComputeStream());
    fmha_buffer = GetScratchBuffer<void>(fmha_buffer_bytes, context->GetComputeStream());

    data.k = reinterpret_cast<CudaT*>(k_buffer.get());
    data.v = reinterpret_cast<CudaT*>(v_buffer.get());
    data.fmha_buffer = reinterpret_cast<CudaT*>(fmha_buffer.get());
  }
#endif

  // -------------
  // Centralized scratch buffer allocation using GQABufferRequirements
  // This ensures allocation logic stays in sync with kernel usage
  auto buffer_req = GQABufferRequirements::Compute<T>(
      parameters,
      data.use_flash_attention,
      data.use_flash_attention_fast_decode,
      data.use_memory_efficient_attention);

  if (buffer_req.qkv_buffer_bytes > 0) {
    unpacked_qkv_buffer = GetScratchBuffer<void>(buffer_req.qkv_buffer_bytes, context->GetComputeStream());
    data.qkv_buffer = reinterpret_cast<CudaT*>(unpacked_qkv_buffer.get());
  }

  if (kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = data.use_flash_attention;
    debug_info.use_efficient_attention = data.use_memory_efficient_attention;

    debug_info.Print("GroupQueryAttention",
                     this->Node().Name(),
                     std::is_same<T, MLFloat16>::value,
                     std::is_same<T, BFloat16>::value);
  }

  // Validate past_value pointer consistency (past_present_share_buffer was computed early after pointer setup)
  if (parameters.past_present_share_buffer) {
    ORT_ENFORCE(data.past_value == data.present_value, "past_value and present_value must be the same tensor when past_present_share_buffer is true");
  } else {
    ORT_ENFORCE(data.past_value != data.present_value, "past_value and present_value must be different tensors when past_present_share_buffer is false");
  }

  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());

  if (parameters.do_rotary) {
    data.cos_cache = reinterpret_cast<const CudaT*>(cos_cache->Data<T>());
    data.sin_cache = reinterpret_cast<const CudaT*>(sin_cache->Data<T>());
  }

  if (head_sink != nullptr) {
    data.head_sink = reinterpret_cast<const CudaT*>(head_sink->Data<T>());
  }

  cublasHandle_t cublas = GetCublasHandle(context);

  ORT_RETURN_IF_ERROR(QkvToContext<CudaT>(
      device_prop, cublas, context->GetComputeStream(), parameters, data));

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

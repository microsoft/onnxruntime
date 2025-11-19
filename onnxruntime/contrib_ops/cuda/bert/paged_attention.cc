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
}

template <typename T>
Status PagedAttention<T>::ComputeInternal(OpKernelContext* context) const {
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

  // Check flash kernel availability and allocate buffers
#if USE_FLASH_ATTENTION
  bool use_flash_attention = !disable_flash_attention_ &&
                             onnxruntime::flash::is_supported(device_prop,
                                                              parameters.head_size,
                                                              parameters.num_heads,
                                                              parameters.kv_num_heads);
  size_t softmax_lse_bytes = 0;
  if (use_flash_attention) {
    softmax_lse_bytes = onnxruntime::flash::get_softmax_lse_size(parameters.token_count,
                                                                 parameters.num_heads);
  }
  auto softmax_lse_buffer = GetScratchBuffer<void>(softmax_lse_bytes, context->GetComputeStream());
#else
  constexpr bool use_flash_attention = false;
  auto softmax_lse_buffer = GetScratchBuffer<void>(0, context->GetComputeStream());  // nullptr
#endif

  if (!use_flash_attention) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Currently PagedAttention is only supported through the FlashAttention kernel.");
  }

  size_t cumulative_seqlens_kv_bytes = sizeof(int) * (parameters.batch_size + 1);
  auto cumulative_seqlens_kv_buffer = GetScratchBuffer<void>(cumulative_seqlens_kv_bytes, context->GetComputeStream());

  size_t workspace_buffer_bytes = 0;
  if (do_rotary_) {
    workspace_buffer_bytes = sizeof(T) * parameters.token_count * (parameters.hidden_size + parameters.kv_hidden_size);
  } else if (parameters.is_packed_qkv) {
    workspace_buffer_bytes = sizeof(T) * parameters.token_count * parameters.hidden_size;
  }
  auto workspace_buffer = GetScratchBuffer<void>(workspace_buffer_bytes, context->GetComputeStream());

  // Print debug info
  if (kernel_options_->AllowDebugInfo()) {
    AttentionKernelDebugInfo debug_info;
    debug_info.use_flash_attention = use_flash_attention;

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
  data.cumulative_seqlens_kv = reinterpret_cast<int*>(cumulative_seqlens_kv_buffer.get());
  data.block_table = reinterpret_cast<const int*>(block_table->Data<int>());
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.use_flash_attention = use_flash_attention;
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

  cublasHandle_t cublas = GetCublasHandle(context);

  return QkvToContext<CudaT>(
      device_prop, cublas, context->GetComputeStream(), parameters, data);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

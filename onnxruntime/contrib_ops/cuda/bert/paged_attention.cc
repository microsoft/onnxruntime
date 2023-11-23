// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/paged_attention.h"
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "contrib_ops/cuda/bert/packed_multihead_attention.h"
#include "contrib_ops/cuda/bert/packed_multihead_attention_impl.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cuda/bert/paged_attention_impl.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/packed_attention_impl.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "gsl/narrow"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// The three struct is used to represent InputMedata in the python side.
struct AttnBias {
  typedef struct {
    int64_t seqstart;
    int64_t max_seqlen;
    int64_t seqstart_py;
  } block_tables;
  block_tables k_seqinfo;
  block_tables q_seqinfo;
  int64_t batchsize;
};

struct THEvent {
  cudaEvent_t events[128];  // assume we have at most 128 layers.
};

struct InputMetadata {
  int64_t schedule_type;  // 0: vllm. 1:sarathi, 2:custom, 3:self-build
  int64_t block_tables;
  int64_t max_num_blocks_per_seq;
  int64_t context_lens;
  int64_t max_context_len;
  int64_t num_prompt_tokens;
  int64_t num_valid_tokens;
  int64_t slot_mapping;
  int64_t num_generation_tokens;
  AttnBias attn_bias;
  THEvent cache_events;
  cudaStream_t cache_stream;
};

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      PagedAttention,                                             \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      PagedAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

namespace at {
#pragma pack(8)
struct Tensor {
  int dtype_;
  const void* data_;
  std::vector<int64_t> shape_;
};
}  // namespace at

#define CHECK_CUDA_ERROR()                                                                                                                  \
  {                                                                                                                                         \
    auto err = cudaGetLastError();                                                                                                          \
    if (err != cudaSuccess) {                                                                                                               \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, " line:", __LINE__, ",  CUDA error ", cudaGetErrorName(err), ":", cudaGetErrorString(err)); \
    }                                                                                                                                       \
  }

#define DEBUG_TENSOR_DUMP 0
void DumpTensor(cudaStream_t cuda_stream, const void* tensor_data, const std::string& name, size_t size_in_bytes) {
#if DEBUG_TENSOR_DUMP
  static std::map<std::string, int> counter;

  std::vector<char> tmp;
  tmp.resize(size_in_bytes);
  CUDA_CALL_THROW(cudaStreamSynchronize(cuda_stream));

  CUDA_CALL_THROW(cudaMemcpy(tmp.data(), tensor_data, size_in_bytes, cudaMemcpyDeviceToHost));
  FILE* file = fopen((name + ".bin." + std::to_string(counter[name])).c_str(), "wb");
  fwrite(tmp.data(), 1, size_in_bytes, file);
  fclose(file);
  counter[name]++;
#else
  ORT_UNUSED_PARAMETER(cuda_stream);
  ORT_UNUSED_PARAMETER(tensor_data);
  ORT_UNUSED_PARAMETER(name);
  ORT_UNUSED_PARAMETER(size_in_bytes);
#endif
}
void DumpTensor(cudaStream_t cuda_stream, const Tensor* tensor, const std::string& name) {
  DumpTensor(cuda_stream, tensor->DataRaw(), name, tensor->SizeInBytes());
}

template <typename T>
AttentionSelector<T>::AttentionSelector(PagedAttention<T>* op) : op_(op) {
#if USE_FLASH_ATTENTION
  disable_flash_attention_ = sizeof(T) != 2 || onnxruntime::ParseEnvironmentVariableWithDefault<bool>(
                                                   attention::kDisableFlashAttention, false);
#else
  disable_flash_attention_ = true;
#endif

  disable_TRT_flash_attention_ = sizeof(T) == 2 &&
                                 ParseEnvironmentVariableWithDefault<bool>(attention::kDisableTrtFlashAttention, false);
  enable_fused_causal_attention_ =
      sizeof(T) == 2 &&
      ParseEnvironmentVariableWithDefault<bool>(attention::kEnableFusedCausalAttention, false);
#if USE_MEMORY_EFFICIENT_ATTENTION
  disable_memory_efficient_attention_ = onnxruntime::ParseEnvironmentVariableWithDefault<bool>(
      attention::kDisableMemoryEfficientAttention, false);
#else
  disable_memory_efficient_attention_ = true;
#endif
}

template <typename T>
SelectResult AttentionSelector<T>::Select(PackedAttentionParameters parameters, const cudaDeviceProp& device_prop) const {
  SelectResult result;
  result.use_flash_attention = false;
#if USE_FLASH_ATTENTION
  if (!disable_flash_attention_) {
    result.use_flash_attention = !parameters.has_relative_position_bias &&
                                 parameters.head_size == parameters.v_head_size &&
                                 onnxruntime::flash::is_supported(device_prop,
                                                                  parameters.head_size,
                                                                  parameters.num_heads,
                                                                  parameters.num_kv_heads);
  }
#endif
  const bool is_unidirectional_ = true;
  result.fused_runner = (result.use_flash_attention ||
                         disable_TRT_flash_attention_ ||
                         !is_unidirectional_ ||
                         !enable_fused_causal_attention_)
                            ? nullptr
                            : op_->GetFusedRunner(device_prop, parameters);

  if (!result.use_flash_attention && nullptr == result.fused_runner && !disable_memory_efficient_attention_) {
    int sm = device_prop.major * 10 + device_prop.minor;
    bool is_good_for_rpb = !parameters.has_relative_position_bias || parameters.sequence_length % (4 * sizeof(T)) == 0;
    result.use_memory_efficient_attention =
        is_good_for_rpb &&
        (sizeof(T) == 2 || parameters.sequence_length >= attention::kMinSeqLenForMemoryEfficientAttentionFp32) &&
        (parameters.head_size & 7) == 0 &&
        (parameters.v_head_size & 7) == 0 &&
        has_memory_efficient_attention(sm, sizeof(T) == 2);
  }

  constexpr size_t element_size = sizeof(T);
  // When the source and target format is same (like TN3H => TN3H, or TNH => TNH) and no bias, need not transpose qkv.
  result.no_qkv_workspace = (result.fused_runner == nullptr) ||
                            ((result.use_memory_efficient_attention || result.use_flash_attention));
  result.workSpaceSize = GetAttentionWorkspaceSize(element_size,
                                                   parameters.batch_size,
                                                   parameters.num_heads,
                                                   parameters.head_size,
                                                   parameters.v_head_size,
                                                   parameters.sequence_length,
                                                   result.fused_runner,
                                                   result.use_flash_attention,
                                                   result.use_memory_efficient_attention,
                                                   result.no_qkv_workspace);

  return result;
}

template <typename T>
PagedAttention<T>::PagedAttention(const OpKernelInfo& info) : CudaKernel(info), selector_(this) {
  int64_t num_heads = 0;
  int64_t head_size = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  int64_t num_kv_heads = info.GetAttrOrDefault("num_kv_heads", num_heads);
  ORT_ENFORCE(info.GetAttr("head_size", &head_size).IsOK() && head_size > 0);
  num_heads_ = static_cast<int32_t>(num_heads);
  num_kv_heads_ = static_cast<int32_t>(num_kv_heads);
  num_queries_per_kv_ = num_heads_ / num_kv_heads_;
  std::vector<int32_t> head_mapping_host(num_heads_);
  for (int i = 0; i < num_kv_heads_; i++) {
    for (int j = 0; j < num_queries_per_kv_; j++) {
      head_mapping_host[i * num_queries_per_kv_ + j] = i;
    }
  }
  auto cuda_mem = GetScratchBuffer<int32_t>(num_heads, 0);
  head_mapping_.swap(cuda_mem);
  CUDA_CALL_THROW(cudaMemcpy(head_mapping_.get(), head_mapping_host.data(), sizeof(int32_t) * num_heads_, cudaMemcpyHostToDevice));

  head_size_ = static_cast<int32_t>(head_size);
  ORT_ENFORCE(info.GetAttr("scale", &scale_).IsOK() && scale_ > 0);
  ORT_ENFORCE(info.GetAttr("mask_type", &mask_type_).IsOK() && (mask_type_ == "normal" || mask_type_ == "alibi" || mask_type_ == "RoPE"));
}

template <typename T>
Status PagedAttention<T>::CheckInputs(
    OpKernelContext* context,
    const InputMetadata* input_metadata,
    PackedAttentionParameters& parameters) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key_cache = context->Input<Tensor>(3);
  const Tensor* value_cache = context->Input<Tensor>(4);
  //const Tensor* positions = context->Input<Tensor>(6);

  const auto& query_shape = query->Shape();
  if (query_shape.NumDimensions() < 2 || query_shape.NumDimensions() > 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid query shape: ", query_shape, " expected 2 or 3 dimensions");
  }
  int64_t batch_size = 1;
  int64_t seq_len = query_shape[0];
  if (query_shape.NumDimensions() == 3) {
    batch_size = query_shape[0];
    seq_len = query_shape[1];
  }

  if (batch_size != 1 && input_metadata->num_prompt_tokens * input_metadata->num_generation_tokens != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Invalid input_medata, batch_size should be 1 when prompt"
                           " and generation tokens are both present");
  }

  if (batch_size * seq_len < input_metadata->num_valid_tokens) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid query shape: ", query_shape,
                           " expected at least ", input_metadata->num_valid_tokens, " tokens");
  }

  if (key_cache->Shape().NumDimensions() != 5 || value_cache->Shape().NumDimensions() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid key_cache or value_cache shape: ",
                           key_cache->Shape(), " ", value_cache->Shape());
  }

  //if (positions && positions->Shape().Size() > 0 &&
  //    positions->Shape()[positions->Shape().NumDimensions() - 1] != batch_size * seq_len) {
  //  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid positions shape: ", positions->Shape());
  //}

  int64_t num_prompt_tokens = input_metadata->num_prompt_tokens;

  // padding removed
  parameters.batch_size = input_metadata->attn_bias.batchsize;
  // parameters.sequence_length = gsl::narrow<int>(num_prompt_tokens);
  parameters.head_size = head_size_;
  parameters.num_heads = num_heads_;
  parameters.num_kv_heads = num_kv_heads_;
  parameters.scale = scale_;

  ////parameters.batch_size = static_cast<int>(batch_size);
  parameters.sequence_length = static_cast<int>(input_metadata->attn_bias.q_seqinfo.max_seqlen);
  parameters.input_hidden_size = -1;  // not applicable
  parameters.hidden_size = static_cast<int>(head_size_ * num_heads_);
  parameters.v_hidden_size = static_cast<int>(head_size_ * num_kv_heads_);
  ////parameters.head_size = static_cast<int>(hidden_size) / num_heads;
  parameters.v_head_size = static_cast<int>(parameters.head_size);
  // parameters.num_heads = num_heads_;
  ////parameters.scale = scale_;
  parameters.token_count = static_cast<int32_t>(num_prompt_tokens);
  parameters.valid_token_count = static_cast<int32_t>(input_metadata->num_valid_tokens);
  parameters.has_relative_position_bias = false;
  parameters.broadcast_res_pos_bias = false;
  parameters.causal = true;

  return Status::OK();
}

template <typename T>
Status PagedAttention<T>::DoQKVProjectionIfNeed(OpKernelContext* context,
                                                InputMetadata* input_metadata,
                                                PackedAttentionParameters parameters,
                                                IAllocatorUniquePtr<T>& gemm_buffer) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);

  // query for input, key for weights, value for bias, so their dimensions are different.
  if (key->Shape().NumDimensions() == value->Shape().NumDimensions()) {
    return Status::OK();
  }
  const auto* input = query;
  const auto* weights = key;
  const auto* bias = value;
  cublasHandle_t cublas = GetCublasHandle(context);

  typedef typename ToCudaType<T>::MappedType CudaT;

  int m = input->Shape()[0];  // input_metadata->num_valid_tokens;
  int n = (parameters.hidden_size + 2 * parameters.v_hidden_size);
  int k = parameters.hidden_size;
  gemm_buffer = GetScratchBuffer<T>(static_cast<size_t>(m) * n, context->GetComputeStream());
  IAllocatorUniquePtr<T> gemm_buffer_pack = GetScratchBuffer<T>(static_cast<size_t>(m) * n, context->GetComputeStream());

  CudaT one = ToCudaType<T>::FromFloat(1.0f);
  CudaT zero = ToCudaType<T>::FromFloat(0.0f);

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrice into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const CudaT*>(weights->Data<T>()), n,
      reinterpret_cast<const CudaT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<CudaT*>(gemm_buffer_pack.get()), n, GetDeviceProp()));

  size_t num_valid_tokens = input_metadata->num_valid_tokens;
  // MQA
  LaunchTranspose<CudaT>(reinterpret_cast<CudaT*>(gemm_buffer_pack.get()), 0, 0,
                         reinterpret_cast<const CudaT*>(bias ? bias->Data<T>() : 0), reinterpret_cast<CudaT*>(gemm_buffer.get()),
                         1, num_valid_tokens,
                         num_heads_, head_size_, head_size_,
                         AttentionQkvFormat::QKV_TN3H, AttentionQkvFormat::Q_K_V_TNH,
                         0, num_valid_tokens, Stream(context));

  CHECK_CUDA_ERROR();
  return Status::OK();
}

template <typename T>
Status PagedAttention<T>::RunMultiHeadAttention(Tensor* output, OpKernelContext* context,
                                                InputMetadata* input_metadata,
                                                PackedAttentionParameters parameters,
                                                IAllocatorUniquePtr<T>& gemm_buffer) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);

  const Tensor* bias = nullptr;
  const Tensor* relative_position_bias = nullptr;
  cublasHandle_t cublas = this->GetCublasHandle(context);
  auto& device_prop = this->GetDeviceProp();

  auto result = selector_.Select(parameters, device_prop);
  if (result.fused_runner == nullptr && result.use_flash_attention == false && result.use_memory_efficient_attention == false) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL,
        "No attention kernel is selected. USE_MEMORY_EFFICIENT_ATTENTION and USE_FLASH_ATTENTION should be enabled.");
  }
  auto work_space = this->template GetScratchBuffer<void>(result.workSpaceSize, context->GetComputeStream());

  int64_t num_valid_tokens = input_metadata->num_valid_tokens;
  typedef typename ToCudaType<T>::MappedType CudaT;
  PackedMultiHeadAttentionData<CudaT> data;
  if (gemm_buffer.get()) {
    data.query = reinterpret_cast<const CudaT*>(gemm_buffer.get());
    data.key = data.query + num_valid_tokens * num_kv_heads_ * head_size_;
    data.value = data.key + num_valid_tokens * num_kv_heads_ * head_size_;
  } else {
    data.query = reinterpret_cast<const CudaT*>(query->Data<T>());
    data.key = reinterpret_cast<const CudaT*>(key->Data<T>());
    data.value = reinterpret_cast<const CudaT*>(value->Data<T>());
  }

  // broadcast key,value for GQA
  TensorShape key_shape({parameters.valid_token_count, parameters.num_kv_heads, parameters.head_size});
  size_t kv_repeat_space = key_shape.Size() * (num_queries_per_kv_ > 0 ? num_queries_per_kv_ : 0);
  IAllocatorUniquePtr<CudaT> key_out = GetScratchBuffer<CudaT>(kv_repeat_space, context->GetComputeStream());
  IAllocatorUniquePtr<CudaT> value_out = GetScratchBuffer<CudaT>(kv_repeat_space, context->GetComputeStream());
  if (num_queries_per_kv_ > 1 && !ParseEnvironmentVariableWithDefault<bool>("repeat_kv_tile", false)) {
    // repeat key and value
    LaunchRepeatKeyValue<CudaT>(Stream(context), key_out.get(), value_out.get(),
                                data.key, data.value, key_shape.GetDims().data(), num_queries_per_kv_);
    CHECK_CUDA_ERROR();
    data.key = key_out.get();
    data.value = value_out.get();
    parameters.num_kv_heads = parameters.num_heads;
    DumpTensor(Stream(context), data.key, "repeat_key", kv_repeat_space * sizeof(CudaT));
  }

  data.bias = (bias == nullptr) ? nullptr : reinterpret_cast<const CudaT*>(bias->Data<T>());
  data.relative_position_bias = (nullptr == relative_position_bias)
                                    ? nullptr
                                    : reinterpret_cast<const CudaT*>(relative_position_bias->Data<T>());
  data.workspace = reinterpret_cast<CudaT*>(work_space.get());
  data.token_offset = nullptr;  // token_offset->Data<int32_t>();
  data.cumulative_sequence_length = reinterpret_cast<int32_t*>(input_metadata->attn_bias.q_seqinfo.seqstart);
  data.output = reinterpret_cast<CudaT*>(output->MutableData<T>());
  data.fused_runner = reinterpret_cast<void*>(result.fused_runner);
  data.use_flash_attention = result.use_flash_attention;
  data.use_memory_efficient_attention = result.use_memory_efficient_attention;
  data.no_qkv_workspace = result.no_qkv_workspace;
  data.source_qkv_format = (key == nullptr) ? AttentionQkvFormat::QKV_TN3H : AttentionQkvFormat::Q_K_V_TNH;
  return QkvToContext<CudaT>(device_prop, cublas, this->Stream(context), parameters, data);
}

InputMetadata* GetOrCreateMedataFromInput(OpKernelContext* context, InputMetadata* s_input_metadata, int8_t* meta_data_space) {
  const Tensor* t_input_metadata = context->Input<Tensor>(5);
  if (t_input_metadata && t_input_metadata->Data<int64_t>()[0]) {
    return reinterpret_cast<InputMetadata*>(t_input_metadata->Data<int64_t>()[0]);
  }

  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key_cache = context->Input<Tensor>(3);
  int seq_len = query->Shape().NumDimensions() == 3 ? query->Shape()[1] : query->Shape()[0];

  const Tensor* positions = context->Input<Tensor>(6);
  std::vector<int64_t> cpu_position(positions->Shape().Size());
  CUDA_CALL_THROW(cudaMemcpy(cpu_position.data(), positions->DataRaw(),
                             positions->SizeInBytes(), cudaMemcpyDeviceToHost));
  while (cpu_position.back() == 0) {
    cpu_position.pop_back();
    seq_len--;
  }
  InputMetadata* input_metadata = s_input_metadata;
  input_metadata->num_valid_tokens = seq_len;

  std::vector<int32_t> slot_mapping;
  std::vector<int32_t> context_lens;
  std::vector<int32_t> block_tables;
  std::vector<int32_t> seqstart;

  input_metadata->max_context_len = 0;
  // in prompt mode
  if (cpu_position.back() == 0 ||
      cpu_position.size() > 1) {
    input_metadata->num_prompt_tokens = seq_len;
    input_metadata->num_generation_tokens = 0;
    slot_mapping.resize(input_metadata->num_prompt_tokens);
    std::iota(slot_mapping.begin(), slot_mapping.end(), 0);
  } else {
    int32_t block_size = gsl::narrow<int32_t>(key_cache->Shape()[3]);
    int32_t past_seq_len = cpu_position.back();
    input_metadata->num_prompt_tokens = 0;
    input_metadata->num_generation_tokens = seq_len;
    slot_mapping.push_back(past_seq_len);
    context_lens.push_back(past_seq_len + 1);
    for (int i = 0; i < past_seq_len + 1; i += block_size) {
      block_tables.push_back(i / block_size);
    }
    input_metadata->max_context_len = context_lens.back();
  }

  if (block_tables.empty()) {
    input_metadata->block_tables = 0;
  } else {
    // copy to cuda
    CUDA_CALL_THROW(cudaMemcpy(meta_data_space, block_tables.data(),
                               block_tables.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    input_metadata->block_tables = reinterpret_cast<int64_t>(meta_data_space);
    meta_data_space += block_tables.size() * sizeof(int32_t);
  }
  if (context_lens.empty()) {
    input_metadata->context_lens = 0;
  } else {
    // copy to cuda
    CUDA_CALL_THROW(cudaMemcpy(meta_data_space, context_lens.data(),
                               context_lens.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    input_metadata->context_lens = reinterpret_cast<int64_t>(meta_data_space);
    meta_data_space += context_lens.size() * sizeof(int32_t);
  }
  {
    // copy to cuda
    CUDA_CALL_THROW(cudaMemcpy(meta_data_space, slot_mapping.data(),
                               slot_mapping.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    input_metadata->slot_mapping = reinterpret_cast<int64_t>(meta_data_space);
    meta_data_space += slot_mapping.size() * sizeof(int32_t);
  }
  input_metadata->max_num_blocks_per_seq = block_tables.size();
  std::memset(input_metadata->cache_events.events, 0, sizeof(THEvent));

  if (input_metadata->num_prompt_tokens > 0) {
    seqstart.push_back(0);
    seqstart.push_back(input_metadata->num_prompt_tokens);

    // copy to cuda
    CUDA_CALL_THROW(cudaMemcpy(meta_data_space, seqstart.data(),
                               seqstart.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    input_metadata->attn_bias.q_seqinfo.seqstart = reinterpret_cast<int64_t>(meta_data_space);
    input_metadata->attn_bias.q_seqinfo.max_seqlen = input_metadata->num_prompt_tokens;
    input_metadata->attn_bias.batchsize = 1;
    meta_data_space += seqstart.size() * sizeof(int32_t);
  }

  return input_metadata;
}

template <typename T>
Status PagedAttention<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* key_cache = context->Input<Tensor>(3);
  const Tensor* value_cache = context->Input<Tensor>(4);
  const Tensor* positions = context->Input<Tensor>(6);
  const Tensor* cos_sin_cache = context->Input<Tensor>(7);
  const Tensor* kv_quant_param = (context->InputCount() > 8) ? context->Input<Tensor>(8) : nullptr;

  const auto& query_shape = query->Shape();
  int seq_len = query->Shape().NumDimensions() == 3 ? query->Shape()[1] : query->Shape()[0];

  auto meta_data_space = this->template GetScratchBuffer<int8_t>(std::max(1024, seq_len * 3) * sizeof(int32_t), context->GetComputeStream());
  InputMetadata self_build_input_metadata;
  InputMetadata* input_metadata = GetOrCreateMedataFromInput(context, &self_build_input_metadata, meta_data_space.get());

  PackedAttentionParameters parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(context, input_metadata, parameters));

  IAllocatorUniquePtr<T> gemm_buffer;
  ORT_RETURN_IF_ERROR(DoQKVProjectionIfNeed(context, input_metadata, parameters, gemm_buffer));

  T* query_data = const_cast<T*>(query->Data<T>());
  T* key_data = const_cast<T*>(key->Data<T>());
  T* value_data = const_cast<T*>(value->Data<T>());

  int64_t num_valid_tokens = input_metadata->num_valid_tokens;
  TensorShape output_shape = query_shape;
  if (gemm_buffer.get() == nullptr) {
    ORT_ENFORCE(query_shape[query_shape.NumDimensions() - 1] == num_heads_ * head_size_, "invlaid query shape");
  } else {
    output_shape[output_shape.NumDimensions() - 1] = num_heads_ * head_size_;
    TensorShapeVector new_shape(2);
    // squeeze(1)
    new_shape[0] = output_shape[0];
    new_shape[1] = output_shape[2];
    output_shape = TensorShape(new_shape);

    query_data = (gemm_buffer.get());
    key_data = (gemm_buffer.get()) + num_valid_tokens * num_heads_ * head_size_;
    value_data = (gemm_buffer.get()) + num_valid_tokens * num_heads_ * head_size_ * 2;
  }

  Tensor* output = context->Output(0, output_shape);

  if (mask_type_ == "RoPE") {
    ORT_ENFORCE(positions != nullptr, "RoPE mask requires position input");
    ORT_ENFORCE(cos_sin_cache != nullptr, "RoPE mask requires position input");
    int64_t rot_dim = cos_sin_cache->Shape()[1];
    ORT_ENFORCE(rot_dim == head_size_, "RoPE mask requires position input with shape [seq_len, head_size]");
    rotary_embedding_neox(Stream(context), positions->Data<int64_t>(), static_cast<void*>(query_data),
                          static_cast<void*>(key_data), head_size_, cos_sin_cache->DataRaw(), num_valid_tokens,
                          rot_dim, num_heads_, num_kv_heads_, 1);
    CHECK_CUDA_ERROR();
  }

  int64_t num_prompt_tokens = input_metadata->num_prompt_tokens;
  if (num_prompt_tokens > 0) {
    ORT_RETURN_IF_ERROR(RunMultiHeadAttention(output, context, input_metadata, parameters, gemm_buffer));
    CHECK_CUDA_ERROR();
  }

  int kv_quant_chunk_size = 0;
  int kv_quant_param_dtype = 0;  // fp32
  if (kv_quant_param != nullptr && kv_quant_param->Shape().Size() > 0) {
    ORT_ENFORCE(key_cache && key_cache->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_INT8);
    ORT_ENFORCE(value_cache && value_cache->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_INT8);
    ORT_ENFORCE(query->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16, "Current only support fp16 with quant kv cache");
    if (kv_quant_param->GetElementType() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      kv_quant_param_dtype = 1;  // fp16
    }
    auto kv_quant_param_shape = kv_quant_param->Shape();  // [num_blocks, 2, num_kv_heads, head_size / kv_quant_chunk_size, block_size]
    ORT_ENFORCE(kv_quant_param_shape.NumDimensions() == 5 && kv_quant_param_shape[3] > 0 && head_size_ % kv_quant_param_shape[3] == 0);
    kv_quant_chunk_size = head_size_ / kv_quant_param_shape[3];
    ORT_ENFORCE(kv_quant_chunk_size > 0 && kv_quant_chunk_size % 4 == 0);
  }
  auto key_cache_shape = key_cache->Shape();
  if (num_valid_tokens > 0 && key_cache_shape.Size() > 3) {
    int64_t key_shape_r[3] = {num_valid_tokens, num_kv_heads_, head_size_};
    int64_t value_shape_r[3] = {num_valid_tokens, num_kv_heads_, head_size_};
    int block_size = gsl::narrow<int>(key_cache_shape[3]);
    reshape_and_cache(Stream(context),
                      key_data,
                      value_data,
                      key_cache->DataRaw(),
                      value_cache->DataRaw(),
                      reinterpret_cast<const int32_t*>(input_metadata->slot_mapping),
                      key_shape_r,
                      value_shape_r,
                      block_size,
                      key_cache_shape[4],
                      1,
                      (kv_quant_param != nullptr) ? (void*)kv_quant_param->DataRaw() : (void*)nullptr,
                      kv_quant_chunk_size,
                      kv_quant_param_dtype);
    CHECK_CUDA_ERROR();
  }

  if (input_metadata->cache_events.events[0]) {
    CUDA_CALL_THROW(cudaStreamWaitEvent(input_metadata->cache_stream, input_metadata->cache_events.events[0]));
    std::copy(input_metadata->cache_events.events + 1, input_metadata->cache_events.events + 80, input_metadata->cache_events.events);
  }

  if (input_metadata->num_generation_tokens > 0) {
    constexpr int PARTITION_SIZE = 512;
    int max_num_partitions = ((input_metadata->max_context_len + PARTITION_SIZE - 1)  / PARTITION_SIZE);
    //TODO : Tune this heuristic.
    bool use_v1 = max_num_partitions == 1 || (query_shape[0] * query_shape[1]) > PARTITION_SIZE ||
    (kv_quant_param != nullptr && kv_quant_param->Shape().Size() > 0);
    int64_t generation_qeury_shape[3] = {num_valid_tokens - num_prompt_tokens, num_heads_, head_size_};
    if (true||use_v1){
      paged_attention_v1(Stream(context),
                         output->MutableData<MLFloat16>() + num_prompt_tokens * num_heads_ * head_size_,
                         query_data + num_prompt_tokens * num_heads_ * head_size_,
                         key_cache->DataRaw(),
                         value_cache->DataRaw(),
                         head_mapping_.get(),
                         scale_,
                         reinterpret_cast<const int32_t*>(input_metadata->block_tables),
                         reinterpret_cast<const int32_t*>(input_metadata->context_lens),
                         value_cache->Shape()[3],
                         input_metadata->max_context_len,
                         nullptr,
                         input_metadata->max_num_blocks_per_seq,
                         generation_qeury_shape,
                         num_queries_per_kv_, 1,
                         kv_quant_param ? kv_quant_param->DataRaw() : nullptr,
                         kv_quant_chunk_size,
                         kv_quant_param_dtype);
    } else {
      auto tmp_output = this->template GetScratchBuffer<T>(query_shape.Size() * max_num_partitions * sizeof(T), context->GetComputeStream());
      auto exp_sums = this->template GetScratchBuffer<T>(query_shape[0] * query_shape [1]* max_num_partitions * sizeof(T), context->GetComputeStream());
      auto max_logits = this->template GetScratchBuffer<T>(query_shape[0] * query_shape[1] * max_num_partitions * sizeof(T), context->GetComputeStream());
      paged_attention_v2(Stream(context),
                         output->MutableData<MLFloat16>() + num_prompt_tokens * num_heads_ * head_size_,
                         exp_sums.get(),
                         max_logits.get(),
                         tmp_output.get(),
                         query_data + num_prompt_tokens * num_heads_ * head_size_,
                         key_cache->DataRaw(),
                         value_cache->DataRaw(),
                         head_mapping_.get(),
                         scale_,
                         reinterpret_cast<const int32_t*>(input_metadata->block_tables),
                         reinterpret_cast<const int32_t*>(input_metadata->context_lens),
                         value_cache->Shape()[3],
                         input_metadata->max_context_len,
                         nullptr,
                         input_metadata->max_num_blocks_per_seq,
                         generation_qeury_shape,
                         num_queries_per_kv_, 1);
    }
    CHECK_CUDA_ERROR();
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

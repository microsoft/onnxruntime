// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/rocm/bert/paged_attention.h"
#include <dlfcn.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include <hip/hip_fp16.h>
#include "contrib_ops/rocm/transformers/dump_rocm_tensor.h"
#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/common/status.h"
#include "core/framework/float16.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/rocm/bert/paged_attention_impl.h"
#include "contrib_ops/rocm/bert/attention_impl.h"
#include "contrib_ops/rocm/bert/batched_gemm_softmax_gemm_permute_pipelines.cuh"
#include "gsl/narrow"

using namespace onnxruntime::rocm;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

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
  hipEvent_t events[128];  // assume we have at most 128 layers.
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
  hipStream_t cache_stream;
};

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      PagedAttention,                                             \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kRocmExecutionProvider,                                     \
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

#define DEBUG_TENSOR_DUMP 0
void DumpTensor(hipStream_t cuda_stream, const void* tensor_data, const std::string& name, size_t size_in_bytes) {
#if DEBUG_TENSOR_DUMP
  static std::map<std::string, int> counter;

  std::vector<char> tmp;
  tmp.resize(size_in_bytes);
  HIP_CALL_THROW(cudaStreamSynchronize(cuda_stream));

  HIP_CALL_THROW(hipMemcpy(tmp.data(), tensor_data, size_in_bytes, hipMemcpyDeviceToHost));
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
void DumpTensor(hipStream_t cuda_stream, const Tensor* tensor, const std::string& name) {
  DumpTensor(cuda_stream, tensor->DataRaw(), name, tensor->SizeInBytes());
}

template <typename T>
PagedAttention<T>::PagedAttention(const OpKernelInfo& info) : RocmKernel(info) {
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
  HIP_CALL_THROW(hipMemcpy(head_mapping_.get(), head_mapping_host.data(), sizeof(int32_t) * num_heads_, hipMemcpyHostToDevice));

  head_size_ = static_cast<int32_t>(head_size);
  ORT_ENFORCE(info.GetAttr("scale", &scale_).IsOK() && scale_ > 0);
  ORT_ENFORCE(info.GetAttr("mask_type", &mask_type_).IsOK() && (mask_type_ == "normal" || mask_type_ == "alibi" || mask_type_ == "RoPE"));

  // using HipT = typename ToHipType<T>::MappedType;
  // using AttentionTunableOp = GemmSoftmaxGemmPermuteTunableOp<HipT>;
  // tunable_op_ = std::make_shared<AttentionTunableOp>();
  const std::string lib_path = ParseEnvironmentVariableWithDefault<std::string>("flash_attention_v2", "libflashattn.so");
  void* fd = dlopen(lib_path.c_str(), RTLD_LOCAL | RTLD_NOW);
  char* error_str = dlerror();
  if (error_str) {
    std::cerr << "Failed to load flash_atten lib:" << lib_path << " with error: " << error_str << std::endl;
  }
  flash_attention_v2_kernel_ = dlsym(fd, "mha_varlen_fwd_c");
  error_str = dlerror();
  if (error_str) {
    std::cerr << "Failed to get symbol with error: " << error_str << std::endl;
  }
  assert(flash_attention_v2_kernel_ != nullptr);
  // std::cerr << "Loaded flash_atten lib:" << lib_path << " with kernel: " << reinterpret_cast<uintptr_t>(flash_attention_v2_kernel_) << std::endl;
  // dlclose(fd);
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
#if 0
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
  auto cublas = GetRocblasHandle(context);

  using HipT = typename ToHipType<T>::MappedType;

  int m = input->Shape()[0];  // input_metadata->num_valid_tokens;
  int n = (parameters.hidden_size + 2 * parameters.v_hidden_size);
  int k = parameters.hidden_size;
  gemm_buffer = GetScratchBuffer<T>(static_cast<size_t>(m) * n, context->GetComputeStream());
  IAllocatorUniquePtr<T> gemm_buffer_pack = GetScratchBuffer<T>(static_cast<size_t>(m) * n, context->GetComputeStream());

  HipT one = ToHipType<T>::FromFloat(1.0f);
  HipT zero = ToHipType<T>::FromFloat(0.0f);

  // Gemm, note that CUDA assumes col-major, so result(N, M) = 1 * weights x input + 1 x bias
  // The bias part is not included here since we fuse bias, transpose and output 3 matrice into one cuda kernel.
  CUBLAS_RETURN_IF_ERROR(rocblasGemmHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one,
      reinterpret_cast<const HipT*>(weights->Data<T>()), n,
      reinterpret_cast<const HipT*>(input->Data<T>()), k,
      &zero, reinterpret_cast<HipT*>(gemm_buffer_pack.get()), n, GetDeviceProp()));

  size_t num_valid_tokens = input_metadata->num_valid_tokens;
  // MQA
  LaunchTranspose<HipT>(reinterpret_cast<HipT*>(gemm_buffer_pack.get()), 0, 0,
                         reinterpret_cast<const HipT*>(bias ? bias->Data<T>() : 0), reinterpret_cast<HipT*>(gemm_buffer.get()),
                         1, num_valid_tokens,
                         num_heads_, head_size_, head_size_,
                         AttentionQkvFormat::QKV_TN3H, AttentionQkvFormat::Q_K_V_TNH,
                         0, num_valid_tokens, Stream(context));
#endif
  return Status::OK();
}

// template <typename T>
// Status PagedAttention<T>::RunMultiHeadAttention(Tensor* output, OpKernelContext* context,
//                                                 InputMetadata* input_metadata,
//                                                 PackedAttentionParameters parameters,
//                                                 IAllocatorUniquePtr<T>& gemm_buffer) const {
//   const Tensor* query = context->Input<Tensor>(0);
//   const Tensor* key = context->Input<Tensor>(1);
//   const Tensor* value = context->Input<Tensor>(2);

//   const Tensor* relative_position_bias = nullptr;
//   auto& device_prop = this->GetDeviceProp();

//   using HipT = typename ToHipType<T>::MappedType;
//   GemmSoftmaxGemmPermuteParams<HipT> params;
//   params.tuning_ctx = GetTuningContext();
//   params.stream = context->GetComputeStream();
//   params.handle = GetRocblasHandle(context);
//   params.attention = &parameters;
//   params.device_prop = &device_prop;
//   params.scale = scale_ == 0 ? 1.0f / sqrt(head_size_) : scale_;

//   using AttentionTunableOp = GemmSoftmaxGemmPermuteTunableOp<HipT>;
//   auto workspace_bytes = AttentionTunableOp::GetWorkspaceNumBytes(&parameters);
//   auto workspace = GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());
//   params.workspace_buffer = reinterpret_cast<HipT*>(workspace.get());

//   int64_t num_valid_tokens = input_metadata->num_valid_tokens;
//   if (gemm_buffer.get()) {
//     params.q_buffer = reinterpret_cast<const HipT*>(gemm_buffer.get());
//     params.k_buffer = params.q_buffer + num_valid_tokens * num_kv_heads_ * head_size_;
//     params.v_buffer = params.k_buffer + num_valid_tokens * num_kv_heads_ * head_size_;
//   } else {
//     params.q_buffer = reinterpret_cast<const HipT*>(query->Data<T>());
//     params.k_buffer = reinterpret_cast<const HipT*>(key->Data<T>());
//     params.v_buffer = reinterpret_cast<const HipT*>(value->Data<T>());
//   }

//   // broadcast key,value for GQA
//   TensorShape key_shape = {num_valid_tokens, num_kv_heads_, head_size_};
//   size_t kv_repeat_space = key_shape.Size() * (num_queries_per_kv_ > 0 ? num_queries_per_kv_ : 0);
//   IAllocatorUniquePtr<HipT> key_out = GetScratchBuffer<HipT>(kv_repeat_space, context->GetComputeStream());
//   IAllocatorUniquePtr<HipT> value_out = GetScratchBuffer<HipT>(kv_repeat_space, context->GetComputeStream());
//   if (num_queries_per_kv_ > 1 && !ParseEnvironmentVariableWithDefault<bool>("repeat_kv_tile", false)) {
//     // repeat key and value
//     LaunchRepeatKeyValue<HipT>(Stream(context), key_out.get(), value_out.get(),
//                               params.k_buffer, params.v_buffer, key_shape.GetDims().data(), num_queries_per_kv_);
//     params.k_buffer = key_out.get();
//     params.v_buffer = value_out.get();
//     // parameters.num_kv_heads = parameters.num_heads;
//     DumpTensor(Stream(context), params.k_buffer, "repeat_key", kv_repeat_space * sizeof(HipT));
//   }

//   params.bias_buffer = (nullptr == relative_position_bias)
//                                     ? nullptr
//                                     : reinterpret_cast<const HipT*>(relative_position_bias->Data<T>());
//   params.out_buffer = reinterpret_cast<HipT*>(output->MutableData<T>());

//   return (*std::static_pointer_cast<AttentionTunableOp>(tunable_op_))(&params);
// }

void flash_attention_v2(hipStream_t stream,
                        const Tensor* query, const Tensor* key, const Tensor* value,
                        float* work_space,
                        Tensor* output, const InputMetadata* input_metadata,
                        PackedAttentionParameters params, void* flash_attention_v2_kernel) {
  at::Tensor query_tensor = {1, query->DataRaw(), {input_metadata->num_prompt_tokens, params.num_heads, params.head_size}};
  at::Tensor key_tensor = {1, key->DataRaw(), {input_metadata->num_prompt_tokens, params.num_heads, params.head_size}};
  at::Tensor value_tensor = {1, value->DataRaw(), {input_metadata->num_prompt_tokens, params.num_heads, params.head_size}};

  at::Tensor softmax_lse = {2, work_space, {input_metadata->attn_bias.batchsize, params.num_heads, input_metadata->attn_bias.q_seqinfo.max_seqlen}};
  std::vector<int64_t> out_shape = query_tensor.shape_;
  {
  //auto v=output->Shape().GetDims();
  //out_shape.assign(v.begin(), v.end());
  }

  at::Tensor output_tensor = {1, output->MutableDataRaw(), out_shape};
  at::Tensor cu_seqlens_q = {4, reinterpret_cast<int32_t*>(input_metadata->attn_bias.q_seqinfo.seqstart), {input_metadata->attn_bias.batchsize + 1}};
  typedef void mha_varlen_fwd_c(hipStream_t stream, const at::Tensor& q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                const at::Tensor& k,                       // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                                const at::Tensor& v,                       // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                                at::Tensor& softmax_lse,                   // b x num_heads x max_seqlen
                                at::Tensor& out_,                          // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                const at::Tensor& cu_seqlens_q,            // b+1
                                const at::Tensor& cu_seqlens_k,            // b+1
                                const int max_seqlen_q, const int max_seqlen_k,
                                const float softmax_scale, const bool is_causal);
  mha_varlen_fwd_c* func = reinterpret_cast<mha_varlen_fwd_c*>(flash_attention_v2_kernel);

  (*func)(stream, query_tensor,
          key_tensor,
          value_tensor,
          softmax_lse,
          output_tensor,
          cu_seqlens_q,
          cu_seqlens_q,
          input_metadata->attn_bias.q_seqinfo.max_seqlen,
          input_metadata->attn_bias.q_seqinfo.max_seqlen,
          params.scale, true);
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
  HIP_CALL_THROW(hipMemcpy(cpu_position.data(), positions->DataRaw(),
                             positions->SizeInBytes(), hipMemcpyDeviceToHost));
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
    HIP_CALL_THROW(hipMemcpy(meta_data_space, block_tables.data(),
                               block_tables.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    input_metadata->block_tables = reinterpret_cast<int64_t>(meta_data_space);
    meta_data_space += block_tables.size() * sizeof(int32_t);
  }
  if (context_lens.empty()) {
    input_metadata->context_lens = 0;
  } else {
    // copy to cuda
    HIP_CALL_THROW(hipMemcpy(meta_data_space, context_lens.data(),
                               context_lens.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    input_metadata->context_lens = reinterpret_cast<int64_t>(meta_data_space);
    meta_data_space += context_lens.size() * sizeof(int32_t);
  }
  {
    // copy to cuda
    HIP_CALL_THROW(hipMemcpy(meta_data_space, slot_mapping.data(),
                               slot_mapping.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    input_metadata->slot_mapping = reinterpret_cast<int64_t>(meta_data_space);
    meta_data_space += slot_mapping.size() * sizeof(int32_t);
  }
  input_metadata->max_num_blocks_per_seq = block_tables.size();
  std::memset(input_metadata->cache_events.events, 0, sizeof(THEvent));

  if (input_metadata->num_prompt_tokens > 0) {
    seqstart.push_back(0);
    seqstart.push_back(input_metadata->num_prompt_tokens);

    // copy to cuda
    HIP_CALL_THROW(hipMemcpy(meta_data_space, seqstart.data(),
                               seqstart.size() * sizeof(int32_t), hipMemcpyHostToDevice));
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
  auto stream = context->GetComputeStream();

  const auto& query_shape = query->Shape();
  int seq_len = query->Shape().NumDimensions() == 3 ? query->Shape()[1] : query->Shape()[0];

  auto meta_data_space = this->template GetScratchBuffer<int8_t>(std::max(1024, seq_len * 3) * sizeof(int32_t), stream);
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
  }

  int64_t num_prompt_tokens = input_metadata->num_prompt_tokens;
  if (num_prompt_tokens > 0) {
    // ORT_RETURN_IF_ERROR(RunMultiHeadAttention(output, context, input_metadata, parameters, gemm_buffer));
    auto workspace = GetScratchBuffer<float>(static_cast<size_t>(
          input_metadata->attn_bias.batchsize * parameters.num_heads * input_metadata->attn_bias.q_seqinfo.max_seqlen),
                                               context->GetComputeStream());
    flash_attention_v2(Stream(context),
                        query,
                        key,
                        value,
                        workspace.get(),
                        output,
                        input_metadata,
                        parameters,
                        flash_attention_v2_kernel_);
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
  }

  if (input_metadata->cache_events.events[0]) {
    HIP_CALL_THROW(hipStreamWaitEvent(input_metadata->cache_stream, input_metadata->cache_events.events[0], 0));
    std::copy(input_metadata->cache_events.events + 1, input_metadata->cache_events.events + 80, input_metadata->cache_events.events);
  }

  if (input_metadata->num_generation_tokens > 0) {
    constexpr int PARTITION_SIZE = 512;
    int max_num_partitions = ((input_metadata->max_context_len + PARTITION_SIZE - 1)  / PARTITION_SIZE);
    //TODO : Tune this heuristic.
    bool use_v1 = max_num_partitions == 1 || (query_shape[0] * query_shape[1]) > PARTITION_SIZE ||
    (kv_quant_param != nullptr && kv_quant_param->Shape().Size() > 0);
    int64_t generation_qeury_shape[3] = {num_valid_tokens - num_prompt_tokens, num_heads_, head_size_};
    if (use_v1){
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
      auto tmp_output = this->template GetScratchBuffer<T>(query_shape.Size() * max_num_partitions * sizeof(T), stream);
      auto exp_sums = this->template GetScratchBuffer<T>(query_shape[0] * query_shape [1]* max_num_partitions * sizeof(T), stream);
      auto max_logits = this->template GetScratchBuffer<T>(query_shape[0] * query_shape[1] * max_num_partitions * sizeof(T), stream);
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
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

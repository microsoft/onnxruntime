// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/paged_attention.h"

#include "cute/config.hpp"
#include "cute/numeric/numeric_types.hpp"

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/paged/paged_attention/paged_attention.h"
#include "contrib_ops/cuda/bert/paged/paged_attention/lbp_attention.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL()                                                                    \
  ONNX_OPERATOR_KERNEL_EX(                                                                   \
      PagedAttention,                                                                        \
      kMSDomain,                                                                             \
      1,                                                                                     \
      kCudaExecutionProvider,                                                                \
      (*KernelDefBuilder::Create())                                                          \
          .InputMemoryType(OrtMemTypeCPUInput, 5)                                            \
          .TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16>())                \
          .TypeConstraint("C", BuildKernelDefConstraints<float, MLFloat16, Float8E4M3FN>()), \
      PagedAttention)

REGISTER_KERNEL();

PagedAttention::PagedAttention(const OpKernelInfo& info) : CudaKernel(info) {
  scale_ = info.GetAttrOrDefault<float>("beta", 0);
  ORT_ENFORCE(info.GetAttr<int64_t>("page_size", &page_size_).IsOK());
  ORT_ENFORCE(info.GetAttr<int64_t>("num_heads", &num_heads_).IsOK());
  ORT_ENFORCE(info.GetAttr<int64_t>("num_kv_heads", &num_kv_heads_).IsOK());
  ORT_ENFORCE(info.GetAttr<int64_t>("kv_quant_group_size", &kv_quant_group_size_).IsOK());
}

Status PagedAttention::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* query = ctx->Input<Tensor>(0);
  const Tensor* key_cache = ctx->Input<Tensor>(1);
  const Tensor* value_cache = ctx->Input<Tensor>(2);
  const Tensor* page_table = ctx->Input<Tensor>(3);
  const Tensor* context_lens = ctx->Input<Tensor>(4);
  const Tensor* max_context_len_tensor = ctx->Input<Tensor>(5);
  const Tensor* alibi = ctx->Input<Tensor>(6);
  const Tensor* kv_quant_param = ctx->Input<Tensor>(7);

  int32_t max_context_len = 32768;  // NOTE: when omitted, will cause some scheduling overhead
  if (max_context_len_tensor) {
    max_context_len = max_context_len_tensor->Data<int32_t>()[0];
  }

  auto q_shape = query->Shape();
  int32_t num_seqs = q_shape[0];
  int32_t num_heads = q_shape[1];
  int32_t head_size = q_shape[2];
  ORT_ENFORCE(num_heads == num_heads_);
  int32_t q_stride = num_heads * head_size;

  auto key_cache_shape = key_cache->Shape();
  int32_t num_pages = key_cache_shape[0];
  int32_t packed_page_size = key_cache_shape[1];
  ORT_ENFORCE(packed_page_size == page_size_ * num_kv_heads_ * head_size);

  auto value_cache_shape = value_cache->Shape();
  ORT_ENFORCE(num_pages == value_cache_shape[0]);
  ORT_ENFORCE(packed_page_size == value_cache_shape[1]);

  auto page_table_shape = page_table->Shape();
  ORT_ENFORCE(num_seqs == page_table_shape[0]);
  int32_t max_num_pages_per_seq = page_table_shape[1];

  ORT_ENFORCE(num_seqs == context_lens->Shape()[0]);

  TensorShapeVector output_shape{num_seqs, num_heads, head_size};
  Tensor* output = ctx->Output(0, output_shape);

  auto cuda_stream = static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle());

  auto kv_onnx_type = key_cache->DataType();

  auto dev_props = &GetDeviceProp();

  bool use_lbp = float(num_seqs * num_kv_heads_) / dev_props->multiProcessorCount < 2;
  using kScheduler = paged::DataParallelOutOfPlace;

  if (kv_onnx_type == DataTypeImpl::GetType<float>()) {  // float kernel does not have lbp instances
    const void* scalebias = nullptr;
    paged::launch_paged_attention_kernel(
        cuda_stream, dev_props,
        output->MutableData<float>(),
        query->Data<float>(),
        key_cache->Data<float>(),
        value_cache->Data<float>(),
        scalebias,
        page_table->Data<int32_t>(),
        context_lens->Data<int32_t>(),
        alibi == nullptr ? nullptr : alibi->Data<float>(),
        scale_ == 0.0f ? 1.0f / sqrt(head_size) : scale_,
        num_seqs, num_heads_, num_kv_heads_, head_size, page_size_,
        max_num_pages_per_seq, q_stride, max_context_len);
  } else if (kv_onnx_type == DataTypeImpl::GetType<MLFloat16>()) {
    const void* scalebias = nullptr;
    if (!use_lbp) {
      paged::launch_paged_attention_kernel(
          cuda_stream, dev_props,
          reinterpret_cast<half*>(output->MutableData<MLFloat16>()),
          reinterpret_cast<const half*>(query->Data<MLFloat16>()),
          reinterpret_cast<const half*>(key_cache->Data<MLFloat16>()),
          reinterpret_cast<const half*>(value_cache->Data<MLFloat16>()),
          scalebias,
          page_table->Data<int32_t>(),
          context_lens->Data<int32_t>(),
          alibi == nullptr ? nullptr : alibi->Data<float>(),
          scale_ == 0.0f ? 1.0f / sqrt(head_size) : scale_,
          num_seqs, num_heads_, num_kv_heads_, head_size, page_size_,
          max_num_pages_per_seq, q_stride, max_context_len);
    } else {
      using Kernel = paged::LBPAttentionKernel<half, half, void, kScheduler>;
      void* workspace;
      Kernel::create_workspace(cuda_stream, &workspace, nullptr, num_seqs, num_heads_, num_kv_heads_, head_size, max_context_len);
      Kernel::launch(
          cuda_stream, dev_props,
          workspace,
          reinterpret_cast<half*>(output->MutableData<MLFloat16>()),
          reinterpret_cast<const half*>(query->Data<MLFloat16>()),
          reinterpret_cast<const half*>(key_cache->Data<MLFloat16>()),
          reinterpret_cast<const half*>(value_cache->Data<MLFloat16>()),
          scalebias,
          page_table->Data<int32_t>(),
          context_lens->Data<int32_t>(),
          alibi == nullptr ? nullptr : alibi->Data<float>(),
          scale_ == 0.0f ? 1.0f / sqrt(head_size) : scale_,
          num_seqs, num_heads_, num_kv_heads_, head_size, page_size_,
          max_num_pages_per_seq, q_stride, max_context_len);
      Kernel::destroy_workspace(cuda_stream, workspace, num_seqs, num_heads_, num_kv_heads_, head_size, max_context_len);
    }
  } else if (kv_onnx_type == DataTypeImpl::GetType<Float8E4M3FN>()) {
    using TKV = cute::float_e4m3_t;
    if (!use_lbp) {
      paged::launch_paged_attention_kernel(
          cuda_stream, dev_props,
          reinterpret_cast<half*>(output->MutableData<MLFloat16>()),
          reinterpret_cast<const half*>(query->Data<MLFloat16>()),
          reinterpret_cast<const TKV*>(key_cache->Data<Float8E4M3FN>()),
          reinterpret_cast<const TKV*>(value_cache->Data<Float8E4M3FN>()),
          reinterpret_cast<const half*>(kv_quant_param->Data<MLFloat16>()),
          page_table->Data<int32_t>(),
          context_lens->Data<int32_t>(),
          alibi == nullptr ? nullptr : alibi->Data<float>(),
          scale_ == 0.0f ? 1.0f / sqrt(head_size) : scale_,
          num_seqs, num_heads_, num_kv_heads_, head_size, page_size_,
          max_num_pages_per_seq, q_stride, max_context_len);
    } else {
      using Kernel = paged::LBPAttentionKernel<half, cute::float_e4m3_t, half, kScheduler>;
      void* workspace;
      Kernel::create_workspace(cuda_stream, &workspace, nullptr, num_seqs, num_heads_, num_kv_heads_, head_size, max_context_len);
      Kernel::launch(
          cuda_stream, dev_props,
          workspace,
          reinterpret_cast<half*>(output->MutableData<MLFloat16>()),
          reinterpret_cast<const half*>(query->Data<MLFloat16>()),
          reinterpret_cast<const TKV*>(key_cache->Data<Float8E4M3FN>()),
          reinterpret_cast<const TKV*>(value_cache->Data<Float8E4M3FN>()),
          reinterpret_cast<const half*>(kv_quant_param->Data<MLFloat16>()),
          page_table->Data<int32_t>(),
          context_lens->Data<int32_t>(),
          alibi == nullptr ? nullptr : alibi->Data<float>(),
          scale_ == 0.0f ? 1.0f / sqrt(head_size) : scale_,
          num_seqs, num_heads_, num_kv_heads_, head_size, page_size_,
          max_num_pages_per_seq, q_stride, max_context_len);
      Kernel::destroy_workspace(cuda_stream, workspace, num_seqs, num_heads_, num_kv_heads_, head_size, max_context_len);
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "kv cache datatype not implemented.");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

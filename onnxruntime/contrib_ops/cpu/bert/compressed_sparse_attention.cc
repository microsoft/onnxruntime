// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/deepseek_v4_compression_common.h"

namespace onnxruntime {
namespace contrib {
namespace deepseek_v4_attention_impl {

template <typename T>
class CompressedSparseAttention final : public OpKernel {
 public:
  explicit CompressedSparseAttention(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr("compress_rate", &compress_rate_).IsOK() && compress_rate_ > 0,
                "CompressedSparseAttention: compress_rate must be greater than zero.");
    ORT_ENFORCE(info.GetAttr("rotary_dim", &rotary_dim_).IsOK() && rotary_dim_ > 0 && rotary_dim_ % 2 == 0,
                "CompressedSparseAttention: rotary_dim must be a positive even value.");
    rms_norm_epsilon_ = info.GetAttrOrDefault<float>("rms_norm_epsilon", 1e-6f);
  }

  Status Compute(OpKernelContext* context) const override {
    for (int index = 0; index < 13; ++index) {
      ORT_RETURN_IF_NOT(context->Input<Tensor>(index) != nullptr,
                        "CompressedSparseAttention: all inputs are required.");
    }
    const Tensor& hidden = *context->Input<Tensor>(0);
    const int64_t batch = hidden.Shape()[0];
    const int64_t head_size = context->Input<Tensor>(4)->Shape()[1] / 2;

    OverlapCompressorResult result;
    ORT_RETURN_IF_ERROR(RunOverlapCompressor<T>(
        hidden, *context->Input<Tensor>(1), *context->Input<Tensor>(2), *context->Input<Tensor>(3),
        *context->Input<Tensor>(4), *context->Input<Tensor>(5), *context->Input<Tensor>(6),
        *context->Input<Tensor>(7), *context->Input<Tensor>(8), *context->Input<Tensor>(9),
        *context->Input<Tensor>(10), *context->Input<Tensor>(11), *context->Input<Tensor>(12),
        compress_rate_, rotary_dim_, rms_norm_epsilon_, result));

    const TensorShape entries_shape = EntryOutputShape(batch, result.entry_count, head_size, result.entries_rank4);
    const auto entries = WriteEntryData(result.entries, batch, result.entry_count, head_size, result.entries_rank4);
    for (int output_index : {0, 3}) {
      WriteFloatVector<T>(*context->Output(output_index, entries_shape), entries);
    }
    WriteFloatVector<T>(*context->Output(1, TensorShape({batch, result.pending_count, 2 * head_size})), result.pending_kv);
    WriteFloatVector<T>(*context->Output(2, TensorShape({batch, result.pending_count, 2 * head_size})), result.pending_gate);
    WriteFloatVector<T>(*context->Output(4, TensorShape({batch, compress_rate_, head_size})), result.overlap_kv);
    WriteFloatVector<T>(*context->Output(5, TensorShape({batch, compress_rate_, head_size})), result.overlap_gate);
    return Status::OK();
  }

 private:
  int64_t compress_rate_{};
  int64_t rotary_dim_{};
  float rms_norm_epsilon_{};
};

}  // namespace deepseek_v4_attention_impl

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      CompressedSparseAttention, kMSDomain, 1, T, kCpuExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", {DataTypeImpl::GetTensorType<int32_t>(), \
                                  DataTypeImpl::GetTensorType<int64_t>()}), \
      deepseek_v4_attention_impl::CompressedSparseAttention<T>);

REGISTER_KERNEL(float)
REGISTER_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime

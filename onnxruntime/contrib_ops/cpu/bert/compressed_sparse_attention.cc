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
    entry_capacity_ = info.GetAttrOrDefault<int64_t>("entry_capacity", 0);
  }

  Status Compute(OpKernelContext* context) const override {
    for (int index = 0; index < 13; ++index) {
      ORT_RETURN_IF_NOT(context->Input<Tensor>(index) != nullptr,
                        "CompressedSparseAttention: all inputs 0-12 are required.");
    }
    const Tensor& hidden = *context->Input<Tensor>(0);
    const Tensor* position_ids = context->Input<Tensor>(1);
    const int64_t batch = hidden.Shape()[0];
    const int64_t head_size = context->Input<Tensor>(4)->Shape()[1] / 2;

    const bool fixed_mode = entry_capacity_ > 0;
    CompressorResult result;
    ORT_RETURN_IF_ERROR(RunCompressor<T>(
      hidden, *context->Input<Tensor>(2), *context->Input<Tensor>(3),
        *context->Input<Tensor>(4), *context->Input<Tensor>(5), *context->Input<Tensor>(6),
        *context->Input<Tensor>(7), *context->Input<Tensor>(8), *context->Input<Tensor>(9),
      *context->Input<Tensor>(10), context->Input<Tensor>(11), context->Input<Tensor>(12),
        compress_rate_, rotary_dim_, rms_norm_epsilon_, result,
        entry_capacity_, position_ids));

    if (fixed_mode) {
      const Tensor* past_entries = context->Input<Tensor>(10);
      const Tensor* past_pending_kv = context->Input<Tensor>(8);
      const TensorShape& entries_shape = past_entries->Shape();
      const auto entries_data = WriteEntryData(result.entries, batch, entry_capacity_, head_size, result.entries_rank4);
      for (int output_index : {0, 3}) {
        WriteFloatVector<T>(*context->Output(output_index, entries_shape), entries_data);
      }
      const int64_t pending_width = 2 * head_size;
      std::vector<float> padded_kv(static_cast<size_t>(past_pending_kv->Shape().Size()), 0.0f);
      std::vector<float> padded_gate(padded_kv.size(), 0.0f);
      for (int64_t b = 0; b < batch; ++b) {
        std::copy_n(result.pending_kv.begin() + b * result.pending_count * pending_width,
                    result.pending_count * pending_width,
                    padded_kv.begin() + b * (compress_rate_ - 1) * pending_width);
        std::copy_n(result.pending_gate.begin() + b * result.pending_count * pending_width,
                    result.pending_count * pending_width,
                    padded_gate.begin() + b * (compress_rate_ - 1) * pending_width);
      }
      WriteFloatVector<T>(*context->Output(1, past_pending_kv->Shape()), padded_kv);
      WriteFloatVector<T>(*context->Output(2, past_pending_kv->Shape()), padded_gate);
      WriteFloatVector<T>(*context->Output(4, TensorShape({batch, compress_rate_, head_size})), result.overlap_kv);
      WriteFloatVector<T>(*context->Output(5, TensorShape({batch, compress_rate_, head_size})), result.overlap_gate);
    } else {
      const TensorShape entries_shape = EntryOutputShape(batch, result.entry_count, head_size, result.entries_rank4);
      const auto entries = WriteEntryData(result.entries, batch, result.entry_count, head_size, result.entries_rank4);
      for (int output_index : {0, 3}) {
        WriteFloatVector<T>(*context->Output(output_index, entries_shape), entries);
      }
      WriteFloatVector<T>(*context->Output(1, TensorShape({batch, result.pending_count, 2 * head_size})), result.pending_kv);
      WriteFloatVector<T>(*context->Output(2, TensorShape({batch, result.pending_count, 2 * head_size})), result.pending_gate);
      WriteFloatVector<T>(*context->Output(4, TensorShape({batch, compress_rate_, head_size})), result.overlap_kv);
      WriteFloatVector<T>(*context->Output(5, TensorShape({batch, compress_rate_, head_size})), result.overlap_gate);
    }
    return Status::OK();
  }

 private:
  int64_t compress_rate_{};
  int64_t rotary_dim_{};
  float rms_norm_epsilon_{};
  int64_t entry_capacity_{};
};

}  // namespace deepseek_v4_attention_impl

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      CompressedSparseAttention, kMSDomain, 1, T, kCpuExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", {DataTypeImpl::GetTensorType<int32_t>(), \
                                  DataTypeImpl::GetTensorType<int64_t>()}) \
          .MayInplace(8, 1)                                             \
          .MayInplace(9, 2)                                             \
          .MayInplace(10, 3)                                            \
          .MayInplace(11, 4)                                            \
          .MayInplace(12, 5),                                           \
      deepseek_v4_attention_impl::CompressedSparseAttention<T>);

REGISTER_KERNEL(float)
REGISTER_KERNEL(MLFloat16)

}  // namespace contrib
}  // namespace onnxruntime

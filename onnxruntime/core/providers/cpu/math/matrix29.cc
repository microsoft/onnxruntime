// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/math/gemm.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/math/matrix29_helper.h"
#include "core/common/safeint.h"

namespace onnxruntime {

template <typename T>
class Gemm29 final : public Gemm<T> {
 public:
  explicit Gemm29(const OpKernelInfo& info) : Gemm<T>(info) {}

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc, bool& is_packed,
                 PrePackedWeights* prepacked_weights) override {
    // transA is ignored for vectors, whose rank may not be known at packing time.
    if (this->trans_A_ != CblasNoTrans) {
      is_packed = false;
      return Status::OK();
    }
    return Gemm<T>::PrePack(tensor, input_idx, std::move(alloc), is_packed, prepacked_weights);
  }

  Status Compute(OpKernelContext* ctx) const override {
    const auto& a = *ctx->Input<Tensor>(0);
    const auto* b = this->packed_b_ ? nullptr : ctx->Input<Tensor>(1);
    const auto* c = ctx->Input<Tensor>(2);
    const bool trans_a = this->trans_A_ != CblasNoTrans && a.Shape().NumDimensions() > 1;
    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(PrepareGemm29(a.Shape(), b ? b->Shape() : this->b_shape_,
                                      c ? &c->Shape() : nullptr, trans_a,
                                      this->trans_B_ != CblasNoTrans, helper));
    auto& y = *ctx->Output(0, helper.OutputShape());
    if (y.Shape().Size() == 0) return Status::OK();
    auto* out = y.MutableData<T>();
    auto* tp = ctx->GetOperatorThreadPool();
    const bool use_bias = c && this->beta_ != 0;
    const bool row_bias = use_bias && c->Shape().Size() == helper.N() &&
                          (c->Shape().NumDimensions() == 0 || c->Shape()[c->Shape().NumDimensions() - 1] == helper.N());
    if (use_bias && !row_bias) {
      Broadcaster broadcast(c->Shape().GetDims(), y.Shape().GetDims());
      const size_t span = broadcast.GetSpanSize();
      const bool scalar = broadcast.iterator1_.GetDeltasFront() == 0;
      for (size_t offset = 0; offset < static_cast<size_t>(y.Shape().Size()); offset += span) {
        const auto* source = c->Data<T>() + broadcast.iterator1_.AdvanceBy(span);
        if (scalar) {
          std::fill_n(out + offset, span, *source);
        } else {
          std::copy_n(source, span, out + offset);
        }
      }
    }
    const TensorShape bias_shape = row_bias ? TensorShape({helper.N()}) : TensorShape({helper.M(), helper.N()});
    for (size_t i = 0; i < helper.OutputOffsets().size(); ++i) {
      auto* output = out + helper.OutputOffsets()[i];
      const T* a_data = helper.K() ? a.Data<T>() + helper.LeftOffsets()[i] : nullptr;
      const T* c_data = use_bias ? (row_bias ? c->Data<T>() : output) : nullptr;
      if constexpr (std::is_same_v<T, float>) {
        if (this->packed_b_ && helper.K() != 0) {
          if (row_bias) GemmBroadcastBias(helper.M(), helper.N(), this->beta_, c_data, &bias_shape, output);
          MlasGemm(trans_a ? CblasTrans : CblasNoTrans, helper.M(), helper.N(), helper.K(),
                   this->alpha_, a_data, trans_a ? helper.M() : helper.K(), this->packed_b_.get(),
                   use_bias ? this->beta_ : 0.0f, output, helper.N(), tp,
                   &this->mlas_backend_kernel_selector_config_);
          continue;
        }
      }
      Gemm<T>::ComputeGemm(trans_a ? CblasTrans : CblasNoTrans, this->trans_B_,
                           helper.M(), helper.N(), helper.K(), static_cast<T>(this->alpha_),
                           a_data, b ? b->Data<T>() : nullptr, static_cast<T>(this->beta_),
                           c_data, use_bias ? &bias_shape : nullptr, output, tp,
                           &this->mlas_backend_kernel_selector_config_);
    }
    return Status::OK();
  }
};

template <typename T>
class GroupedMatMul final : public OpKernel {
 public:
  explicit GroupedMatMul(const OpKernelInfo& info) : OpKernel(info) {
    trans_b_ = info.GetAttrOrDefault<int64_t>("transB", 0) != 0;
    SetupMlasBackendKernelSelectorFromConfigOptions(config_, info.GetConfigOptions());
  }

  Status Compute(OpKernelContext* ctx) const override {
    const auto& input = *ctx->Input<Tensor>(0);
    const auto& weights = *ctx->Input<Tensor>(1);
    const auto& indices = *ctx->Input<Tensor>(2);
    const auto* bias = ctx->Input<Tensor>(3);
    GroupedMatMulShape shape;
    ORT_RETURN_IF_ERROR(shape.Compute(input.Shape(), weights.Shape(), indices.Shape(),
                                      bias ? &bias->Shape() : nullptr, trans_b_));
    const size_t count = narrow<size_t>(indices.Shape().Size());
    auto index = [&](size_t i) -> int64_t {
      return indices.IsDataType<int32_t>() ? indices.Data<int32_t>()[i] : indices.Data<int64_t>()[i];
    };
    for (size_t i = 0; i < count; ++i) {
      const int64_t group = index(i);
      ORT_RETURN_IF_NOT(group >= 0 && group < shape.g,
                        "GroupedMatMul group_indices[", i, "]=", group, " is outside [0,", shape.g,
                        "). Supply a valid nonnegative group index.");
    }
    auto& output = *ctx->Output(0, {shape.m, shape.s, shape.n});
    if (output.Shape().Size() == 0) return Status::OK();
    auto* out = output.MutableData<T>();
    if (shape.k == 0) {
      for (size_t i = 0; i < count; ++i) {
        if (bias) {
          std::copy_n(bias->Data<T>() + index(i) * shape.n, shape.n, out + i * shape.n);
        } else {
          std::fill_n(out + i * shape.n, shape.n, T{});
        }
      }
      return Status::OK();
    }
    InlinedVector<size_t> offsets(narrow<size_t>(shape.g) + 1, 0);
    for (size_t i = 0; i < count; ++i) ++offsets[static_cast<size_t>(index(i)) + 1];
    for (size_t g = 1; g < offsets.size(); ++g) offsets[g] += offsets[g - 1];
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
    auto packed_input = IAllocator::MakeUniquePtr<T>(allocator, SafeInt<size_t>(count) * shape.k);
    auto packed_output = IAllocator::MakeUniquePtr<T>(allocator, SafeInt<size_t>(count) * shape.n);
    InlinedVector<size_t> order(count);
    auto cursor = offsets;
    // Pack tokens, not weights: repeated selections reuse each group's original weight matrix.
    for (size_t i = 0; i < count; ++i) {
      const size_t position = cursor[static_cast<size_t>(index(i))]++;
      order[position] = i;
      std::copy_n(input.Data<T>() + (i / shape.s) * shape.k, shape.k,
                  packed_input.get() + position * shape.k);
    }
    const TensorShape bias_shape({shape.n});
    for (size_t g = 0; g + 1 < offsets.size(); ++g) {
      const size_t rows = offsets[g + 1] - offsets[g];
      if (rows == 0) continue;
      Gemm<T>::ComputeGemm(CblasNoTrans, trans_b_ ? CblasTrans : CblasNoTrans, rows, shape.n, shape.k,
                           static_cast<T>(1.0f), packed_input.get() + offsets[g] * shape.k,
                           weights.Data<T>() + g * shape.k * shape.n, static_cast<T>(1.0f),
                           bias ? bias->Data<T>() + g * shape.n : nullptr, bias ? &bias_shape : nullptr,
                           packed_output.get() + offsets[g] * shape.n, ctx->GetOperatorThreadPool(), &config_);
    }
    for (size_t position = 0; position < count; ++position) {
      std::copy_n(packed_output.get() + position * shape.n, shape.n, out + order[position] * shape.n);
    }
    return Status::OK();
  }

 private:
  bool trans_b_;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config_{};
};

#define REGISTER_MATRIX29(T)                                                                                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(Gemm, 29, T,                                                                \
                                 KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),   \
                                 Gemm29<T>);                                                                 \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(GroupedMatMul, 29, T,                                                       \
                                 KernelDefBuilder()                                                          \
                                     .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                  \
                                     .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()), \
                                 GroupedMatMul<T>);

REGISTER_MATRIX29(float)
REGISTER_MATRIX29(double)
REGISTER_MATRIX29(MLFloat16)

}  // namespace onnxruntime

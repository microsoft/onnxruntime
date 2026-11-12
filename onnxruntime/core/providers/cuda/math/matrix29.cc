// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/math/matmul.h"
#include "core/providers/cuda/math/matrix29_impl.h"
#include "core/providers/cpu/math/matrix29_helper.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class Gemm29 final : public CudaKernel {
 public:
  explicit Gemm29(const OpKernelInfo& info) : CudaKernel(info) {
    trans_a_ = info.GetAttrOrDefault<int64_t>("transA", 0) != 0;
    trans_b_ = info.GetAttrOrDefault<int64_t>("transB", 0) != 0;
    alpha_ = info.GetAttrOrDefault<float>("alpha", 1.0f);
    beta_ = info.GetAttrOrDefault<float>("beta", 1.0f);
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    using CudaT = typename ToCudaType<T>::MappedType;
    const auto& a = *ctx->Input<Tensor>(0);
    const auto& b = *ctx->Input<Tensor>(1);
    const auto* c = ctx->Input<Tensor>(2);
    MatMulComputeHelper helper;
    ORT_RETURN_IF_ERROR(PrepareGemm29(a.Shape(), b.Shape(), c ? &c->Shape() : nullptr, trans_a_, trans_b_, helper));
    auto& y = *ctx->Output(0, helper.OutputShape());
    const int64_t count = y.Shape().Size();
    if (count == 0) return Status::OK();
    // cuBLAS uses 32-bit dimensions, even though tensor sizes and batch strides are 64-bit.
    ORT_RETURN_IF_NOT(helper.M() <= INT_MAX && helper.N() <= INT_MAX && helper.K() <= INT_MAX &&
                          helper.OutputOffsets().size() <= INT_MAX,
                      "Gemm 29 matrix dimensions or batch count exceed cuBLAS INT_MAX. Split the input into smaller batches.");
    auto* out = reinterpret_cast<CudaT*>(y.MutableData<T>());
    if (helper.K() == 0) {
      Fill<CudaT>(Stream(ctx), out, ToCudaType<T>::FromFloat(alpha_ * 0.0f), count);
    } else {
      ORT_RETURN_IF_ERROR(FuncMatMul<T>(this, ctx, &a, &b, alpha_, trans_a_, trans_b_, false, false, &y));
    }
    if (!c || beta_ == 0) return Status::OK();
    const auto* bias_data = reinterpret_cast<const CudaT*>(c->Data<T>());
    if (c->Shape().Size() == 1 || c->Shape().Size() == count) {
      return Gemm29AddBias(Stream(ctx), out, bias_data, beta_, count,
                           c->Shape().Size() == 1 ? 0 : -1, nullptr, nullptr);
    }
    const size_t rank = c->Shape().NumDimensions();
    CudaAsyncBuffer<int64_t> metadata(this, rank * 2);
    auto values = metadata.CpuSpan();
    int64_t stride = 1;
    const size_t padding = y.Shape().NumDimensions() - rank;
    for (size_t d = rank; d-- > 0;) {
      values[d] = y.Shape()[padding + d];
      values[rank + d] = c->Shape()[d] == 1 ? 0 : stride;
      stride = SafeInt<int64_t>(stride) * c->Shape()[d];
    }
    ORT_RETURN_IF_ERROR(metadata.CopyToGpu(GetComputeStream(ctx)));
    return Gemm29AddBias(Stream(ctx), out, bias_data, beta_, count,
                         narrow<int>(rank), metadata.GpuPtr(), metadata.GpuPtr() + rank);
  }

 private:
  bool trans_a_, trans_b_;
  float alpha_, beta_;
};

template <typename T>
class GroupedMatMul final : public CudaKernel {
 public:
  explicit GroupedMatMul(const OpKernelInfo& info) : CudaKernel(info) {
    trans_b_ = info.GetAttrOrDefault<int64_t>("transB", 0) != 0;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    using CudaT = typename ToCudaType<T>::MappedType;
    const auto& input = *ctx->Input<Tensor>(0);
    const auto& weights = *ctx->Input<Tensor>(1);
    const auto& indices = *ctx->Input<Tensor>(2);
    const auto* bias = ctx->Input<Tensor>(3);
    GroupedMatMulShape shape;
    ORT_RETURN_IF_ERROR(shape.Compute(input.Shape(), weights.Shape(), indices.Shape(),
                                      bias ? &bias->Shape() : nullptr, trans_b_));
    const size_t count = narrow<size_t>(indices.Shape().Size());
    ORT_RETURN_IF_NOT(count <= INT_MAX && shape.n <= INT_MAX && shape.k <= INT_MAX,
                      "GroupedMatMul dimensions exceed cuBLAS INT_MAX. Split tokens or selections into smaller batches.");
    auto index = [&](size_t i) -> int64_t {
      return indices.IsDataType<int32_t>() ? indices.Data<int32_t>()[i] : indices.Data<int64_t>()[i];
    };
    for (size_t i = 0; i < count; ++i) {
      const auto group = index(i);
      ORT_RETURN_IF_NOT(group >= 0 && group < shape.g,
                        "GroupedMatMul group_indices[", i, "]=", group, " is outside [0,", shape.g,
                        "). Supply a valid nonnegative group index.");
    }
    auto& output = *ctx->Output(0, {shape.m, shape.s, shape.n});
    if (output.Shape().Size() == 0) return Status::OK();
    auto* out = reinterpret_cast<CudaT*>(output.MutableData<T>());
    const auto* x = reinterpret_cast<const CudaT*>(input.Data<T>());
    const auto* w = reinterpret_cast<const CudaT*>(weights.Data<T>());
    if (count > 32 && shape.k != 0) {
      InlinedVector<size_t> offsets(narrow<size_t>(shape.g) + 1, 0);
      for (size_t i = 0; i < count; ++i) ++offsets[static_cast<size_t>(index(i)) + 1];
      for (size_t g = 1; g < offsets.size(); ++g) offsets[g] += offsets[g - 1];
      auto cursor = offsets;
      CudaAsyncBuffer<int64_t> order(this, count);
      CudaAsyncBuffer<const CudaT*> selected_bias(this, bias ? count : 0);
      for (size_t i = 0; i < count; ++i) {
        const size_t group = static_cast<size_t>(index(i));
        const size_t position = cursor[group]++;
        order.CpuSpan()[position] = narrow<int64_t>(i);
        if (bias) selected_bias.CpuSpan()[position] = reinterpret_cast<const CudaT*>(bias->Data<T>()) + group * shape.n;
      }
      ORT_RETURN_IF_ERROR(order.CopyToGpu(GetComputeStream(ctx)));
      if (bias) ORT_RETURN_IF_ERROR(selected_bias.CopyToGpu(GetComputeStream(ctx)));
      const size_t input_count = SafeInt<size_t>(count) * shape.k;
      auto packed_input = GetScratchBuffer<CudaT>(input_count, GetComputeStream(ctx));
      auto packed_output = GetScratchBuffer<CudaT>(narrow<size_t>(output.Shape().Size()), GetComputeStream(ctx));
      ORT_RETURN_IF_ERROR(GroupedMatMulGatherRows(Stream(ctx), x, packed_input.get(), order.GpuPtr(),
                                                  narrow<int64_t>(input_count), shape.k, shape.s));
      const CudaT one = ToCudaType<T>::FromFloat(1.0f);
      const CudaT zero = ToCudaType<T>::FromFloat(0.0f);
      // Prefill uses one GEMM per group so tokens share weight reads; small decode batches
      // below use pointer-batched GEMV to avoid one host launch for every selected group.
      for (size_t group = 0; group + 1 < offsets.size(); ++group) {
        const size_t rows = offsets[group + 1] - offsets[group];
        if (rows == 0) continue;
        CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
            GetCublasHandle(ctx), trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
            narrow<int>(shape.n), narrow<int>(rows), narrow<int>(shape.k), &one,
            w + group * shape.k * shape.n, narrow<int>(trans_b_ ? shape.k : shape.n),
            packed_input.get() + offsets[group] * shape.k, narrow<int>(shape.k), &zero,
            packed_output.get() + offsets[group] * shape.n, narrow<int>(shape.n), GetDeviceProp(),
            UseTF32() && shape.k % 4 == 0 && shape.n % 4 == 0));
      }
      return GroupedMatMulScatterRows(Stream(ctx), packed_output.get(), out, order.GpuPtr(),
                                      bias ? selected_bias.GpuPtr() : nullptr, output.Shape().Size(), shape.n);
    }
    CudaAsyncBuffer<const CudaT*> x_array(this, count), w_array(this, count), bias_array(this, bias ? count : 0);
    CudaAsyncBuffer<CudaT*> y_array(this, count);
    for (size_t i = 0; i < count; ++i) {
      if (shape.k != 0) {
        x_array.CpuSpan()[i] = x + (i / shape.s) * shape.k;
        w_array.CpuSpan()[i] = w + index(i) * shape.k * shape.n;
      }
      y_array.CpuSpan()[i] = out + i * shape.n;
      if (bias) bias_array.CpuSpan()[i] = reinterpret_cast<const CudaT*>(bias->Data<T>()) + index(i) * shape.n;
    }
    if (bias) ORT_RETURN_IF_ERROR(bias_array.CopyToGpu(GetComputeStream(ctx)));
    if (bias || shape.k == 0) {
      ORT_RETURN_IF_ERROR(GroupedMatMulBias(Stream(ctx), out, bias ? bias_array.GpuPtr() : nullptr,
                                            output.Shape().Size(), shape.n));
    }
    if (shape.k == 0) return Status::OK();
    ORT_RETURN_IF_ERROR(x_array.CopyToGpu(GetComputeStream(ctx)));
    ORT_RETURN_IF_ERROR(w_array.CopyToGpu(GetComputeStream(ctx)));
    ORT_RETURN_IF_ERROR(y_array.CopyToGpu(GetComputeStream(ctx)));
    const CudaT one = ToCudaType<T>::FromFloat(1.0f);
    const CudaT beta = ToCudaType<T>::FromFloat(bias ? 1.0f : 0.0f);
    CUBLAS_RETURN_IF_ERROR(cublasGemmBatchedHelper(
        GetCublasHandle(ctx), trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
        narrow<int>(shape.n), 1, narrow<int>(shape.k), &one, w_array.GpuPtr(),
        narrow<int>(trans_b_ ? shape.k : shape.n), x_array.GpuPtr(), narrow<int>(shape.k), &beta,
        y_array.GpuPtr(), narrow<int>(shape.n), narrow<int>(count), GetDeviceProp(),
        UseTF32() && shape.k % 4 == 0 && shape.n % 4 == 0));
    return Status::OK();
  }

 private:
  bool trans_b_;
};

#define REGISTER_MATRIX29(T)                                                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Gemm, kOnnxDomain, 29, T, kCudaExecutionProvider,                           \
                                (*KernelDefBuilder::Create())                                               \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),                 \
                                Gemm29<T>);                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(GroupedMatMul, kOnnxDomain, 29, T, kCudaExecutionProvider,                  \
                                (*KernelDefBuilder::Create())                                               \
                                    .InputMemoryType(OrtMemTypeCPUInput, 2)                                 \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                  \
                                    .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()), \
                                GroupedMatMul<T>);

REGISTER_MATRIX29(float)
REGISTER_MATRIX29(double)
REGISTER_MATRIX29(MLFloat16)
REGISTER_MATRIX29(BFloat16)

}  // namespace cuda
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_FLOAT8_TYPES)

#include "contrib_ops/cpu/quantization/gather_quantized.h"

#include <unordered_map>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/common/float16.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

template <typename T1, typename Tind>
Status GatherQuantized<T1, Tind>::PrepareForCompute(OpKernelContext* context, Prepare& p) const {
  p.data_tensor = context->Input<Tensor>(0);
  p.indices_tensor = context->Input<Tensor>(1);
  p.scales_tensor = context->Input<Tensor>(2);

  const auto& data_shape = p.data_tensor->Shape();
  const auto data_rank = data_shape.NumDimensions();
  ORT_RETURN_IF_NOT(data_rank > 1, "data tensor must have rank > 1.");

  p.gather_axis = HandleNegativeAxis(gather_axis_, narrow<int64_t>(data_rank));
  p.quantize_axis = HandleNegativeAxis(quantize_axis_, narrow<int64_t>(data_rank));

  const auto& indices_shape = p.indices_tensor->Shape();
  const auto indices_rank = indices_shape.NumDimensions();

  std::vector<int64_t> shape;
  shape.reserve(data_rank - 1 + indices_rank);

  // get output tensor
  // replace the dimension for p.gather_axis with the shape from the indices
  for (int64_t i = 0; i < p.gather_axis; ++i)
    shape.push_back(data_shape[narrow<size_t>(i)]);

  for (const auto dim : indices_shape.GetDims())
    shape.push_back(dim);

  for (int64_t i = p.gather_axis + 1; i < static_cast<int64_t>(data_rank); ++i)
    shape.push_back(data_shape[narrow<size_t>(i)]);

  p.output_tensor = context->Output(0, TensorShape(std::move(shape)));

  // validate scale shape
  const auto& scales_shape = p.scales_tensor->Shape();
  ORT_RETURN_IF_NOT(data_shape.NumDimensions() == scales_shape.NumDimensions(),
                    "data and scales must have the same rank.");

  const int64_t quantize_axis_dim = data_shape[narrow<size_t>(p.quantize_axis)];
  const int64_t effective_block_size = block_size_ == 0 ? quantize_axis_dim : block_size_;
  for (size_t i = 0; i < data_shape.NumDimensions(); ++i) {
    ORT_RETURN_IF_NOT(i == static_cast<size_t>(p.quantize_axis)
                          ? (data_shape[i] + effective_block_size - 1) / effective_block_size == scales_shape[i]
                          : data_shape[i] == scales_shape[i],
                      "data and scales do not match shapes.");
  }

  return Status::OK();
}

template <typename T1, typename Tind>
template <typename T2>
Status GatherQuantized<T1, Tind>::CopyDataAndDequantize(const T1* data_ptr,
                                                        const Tind* indices_ptr,
                                                        const T2* scales_ptr,
                                                        T2* output_ptr,
                                                        int64_t gather_M,
                                                        int64_t gather_N,
                                                        int64_t gather_axis_dim,
                                                        int64_t gather_block,
                                                        int64_t quantize_axis_dim,
                                                        int64_t quantize_N,
                                                        int64_t effective_block_size,
                                                        concurrency::ThreadPool* tp) const {
  auto data_full_block = gather_axis_dim * gather_block;
  auto quantize_full_block = quantize_axis_dim * quantize_N;
  auto scale_full_block = (quantize_axis_dim + effective_block_size - 1) / effective_block_size * quantize_N;

  auto lambda = [&](int64_t gather_MN_idx) {
    int64_t gather_M_idx = gather_MN_idx / gather_N;
    int64_t gather_N_idx = gather_MN_idx % gather_N;

    int64_t indices_val = static_cast<int64_t>(indices_ptr[gather_N_idx]);
    ORT_ENFORCE(indices_val >= -gather_axis_dim && indices_val < gather_axis_dim,
                "indices element out of data bounds, idx=", indices_val,
                " must be within the inclusive range [", -gather_axis_dim, ",", gather_axis_dim - 1, "]");

    indices_val = indices_val < 0 ? indices_val + gather_axis_dim : indices_val;
    int64_t output_idx_base = gather_MN_idx * gather_block;
    int64_t data_idx_base = gather_M_idx * data_full_block + indices_val * gather_block;

    int64_t output_idx = output_idx_base;
    int64_t data_idx = data_idx_base;
    for (int64_t i = 0; i < gather_block; ++i, ++output_idx, ++data_idx) {
      const float data_val = data_ptr[data_idx].ToFloat();

      int64_t x = data_idx / quantize_full_block;
      int64_t y = data_idx % quantize_full_block / quantize_N;
      int64_t z = data_idx % quantize_N;
      int64_t scale_idx = x * scale_full_block + y / effective_block_size * quantize_N + z;
      const float scale_val = static_cast<float>(scales_ptr[scale_idx]);

      output_ptr[output_idx] = static_cast<T2>(data_val * scale_val);
    }
  };

  concurrency::ThreadPool::TryParallelFor(
      tp,
      SafeInt<ptrdiff_t>(gather_M) * gather_N,
      static_cast<double>(gather_block * 2),
      [&lambda](ptrdiff_t first, ptrdiff_t last) {
        for (auto index = static_cast<int64_t>(first), end = static_cast<int64_t>(last);
             index < end;
             ++index) {
          lambda(index);
        }
      });

  return Status::OK();
}

template <typename T1, typename Tind>
Status GatherQuantized<T1, Tind>::Compute(OpKernelContext* context) const {
  Prepare p;
  ORT_RETURN_IF_ERROR(PrepareForCompute(context, p));
  const auto& data_shape = p.data_tensor->Shape();

  // re-shape the data tensor to [gather_M, gather_axis_dim, gather_block]
  // re-shape the indices tensor to [gather_N]
  // re-shape the output tensor to [gather_M, gather_N, gather_block]
  const int64_t gather_block = data_shape.SizeFromDimension(SafeInt<size_t>(p.gather_axis) + 1);
  const int64_t gather_axis_dim = data_shape[narrow<size_t>(p.gather_axis)];
  const int64_t gather_M = data_shape.SizeToDimension(narrow<size_t>(p.gather_axis));
  const int64_t gather_N = p.indices_tensor->Shape().Size();

  const int64_t quantize_axis_dim = data_shape[narrow<size_t>(p.quantize_axis)];
  const int64_t quantize_N = data_shape.SizeFromDimension(SafeInt<size_t>(p.quantize_axis) + 1);
  const int64_t effective_block_size = block_size_ == 0 ? quantize_axis_dim : block_size_;

  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  const auto* data_ptr = p.data_tensor->template Data<T1>();
  const auto* indices_ptr = p.indices_tensor->template Data<Tind>();
  const auto dequantized_type = p.scales_tensor->GetElementType();

  if (dequantized_type == ONNX_NAMESPACE::TensorProto::FLOAT) {
    const auto* scales_ptr = p.scales_tensor->template Data<float>();
    auto* output_ptr = p.output_tensor->template MutableData<float>();

    return CopyDataAndDequantize<float>(data_ptr, indices_ptr, scales_ptr, output_ptr, gather_M, gather_N,
                                        gather_axis_dim, gather_block, quantize_axis_dim, quantize_N,
                                        effective_block_size, tp);
  } else if (dequantized_type == ONNX_NAMESPACE::TensorProto::FLOAT16) {
    const auto* scales_ptr = p.scales_tensor->template Data<MLFloat16>();
    auto* output_ptr = p.output_tensor->template MutableData<MLFloat16>();

    return CopyDataAndDequantize<MLFloat16>(data_ptr, indices_ptr, scales_ptr, output_ptr, gather_M, gather_N,
                                            gather_axis_dim, gather_block, quantize_axis_dim, quantize_N,
                                            effective_block_size, tp);
  } else if (dequantized_type == ONNX_NAMESPACE::TensorProto::BFLOAT16) {
    const auto* scales_ptr = p.scales_tensor->template Data<BFloat16>();
    auto* output_ptr = p.output_tensor->template MutableData<BFloat16>();

    return CopyDataAndDequantize<BFloat16>(data_ptr, indices_ptr, scales_ptr, output_ptr, gather_M, gather_N,
                                           gather_axis_dim, gather_block, quantize_axis_dim, quantize_N,
                                           effective_block_size, tp);
  } else {
    ORT_THROW("Unsupported dequantized type: ", dequantized_type);
  }
}

#define REGISTER_GATHERQUANTIZED(T1, Tind)                                                                        \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                                                              \
      GatherQuantized,                                                                                            \
      kMSDomain, 1,                                                                                               \
      T1, Tind,                                                                                                   \
      kCpuExecutionProvider,                                                                                      \
      KernelDefBuilder()                                                                                          \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())                                                \
          .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(),                                            \
                                 DataTypeImpl::GetTensorType<MLFloat16>(),                                         \
                                 DataTypeImpl::GetTensorType<BFloat16>()})                                        \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tind>()),                                           \
      GatherQuantized<T1, Tind>);

REGISTER_GATHERQUANTIZED(Float8E4M3FN, int32_t);
REGISTER_GATHERQUANTIZED(Float8E4M3FN, int64_t);
REGISTER_GATHERQUANTIZED(Float8E4M3FNUZ, int32_t);
REGISTER_GATHERQUANTIZED(Float8E4M3FNUZ, int64_t);
REGISTER_GATHERQUANTIZED(Float8E5M2, int32_t);
REGISTER_GATHERQUANTIZED(Float8E5M2, int64_t);
REGISTER_GATHERQUANTIZED(Float8E5M2FNUZ, int32_t);
REGISTER_GATHERQUANTIZED(Float8E5M2FNUZ, int64_t);

}  // namespace contrib
}  // namespace onnxruntime

#endif  // !defined(DISABLE_FLOAT8_TYPES)

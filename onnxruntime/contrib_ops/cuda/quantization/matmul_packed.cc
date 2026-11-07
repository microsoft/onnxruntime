// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_packed.h"

#include <type_traits>

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_MATMUL_PACKED_CUDA(T)                                     \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                           \
      MatMulPacked,                                                        \
      kMSDomain, 1,                                                        \
      T,                                                                   \
      kCudaExecutionProvider,                                              \
      (*KernelDefBuilder::Create())                                        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())          \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>())    \
          .TypeConstraint("Tscale", DataTypeImpl::GetTensorType<float>()), \
      MatMulPacked<T>);

REGISTER_MATMUL_PACKED_CUDA(MLFloat16)
REGISTER_MATMUL_PACKED_CUDA(BFloat16)

#undef REGISTER_MATMUL_PACKED_CUDA

template <typename T>
MatMulPacked<T>::MatMulPacked(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("K", &k_).IsOK() && k_ > 0, "MatMulPacked requires K > 0");
  ORT_ENFORCE(info.GetAttr<int64_t>("N", &n_).IsOK() && n_ > 0, "MatMulPacked requires N > 0");
  ORT_ENFORCE(info.GetAttr<int64_t>("packed_type", &ggml_type_).IsOK() && ggml_type_ >= 0,
              "MatMulPacked requires packed_type >= 0");
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(ggml_type_) || ggml_type_ == gguf_quant::kGgmlTypeNVFP4,
              "MatMulPacked: unsupported packed_type=", ggml_type_);
  const int64_t round_weights = info.GetAttrOrDefault<int64_t>("dequantize_to_activation_dtype", 0);
  ORT_ENFORCE(round_weights == 0 || round_weights == 1, "dequantize_to_activation_dtype must be 0 or 1");
  dequantize_to_activation_dtype_ = round_weights != 0;
}

template <typename T>
Status MatMulPacked<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* a = context->Input<Tensor>(0);
  const Tensor* b = context->Input<Tensor>(1);
  const Tensor* global_scale = context->InputCount() > 2 ? context->Input<Tensor>(2) : nullptr;
  ORT_RETURN_IF_NOT(a != nullptr && b != nullptr, "MatMulPacked requires inputs A and B");
  ORT_RETURN_IF_NOT(!global_scale ||
                        (ggml_type_ == gguf_quant::kGgmlTypeNVFP4 &&
                         (global_scale->Shape().NumDimensions() == 0 ||
                          global_scale->Shape() == TensorShape({1}))),
                    "MatMulPacked global_scale requires NVFP4 and a scalar or [1] shape");

  const auto& a_shape = a->Shape();
  ORT_RETURN_IF_NOT(a_shape.NumDimensions() >= 1, "MatMulPacked input A must have rank >= 1");
  const size_t last_axis = a_shape.NumDimensions() - 1;
  ORT_RETURN_IF_NOT(a_shape[last_axis] == k_, "MatMulPacked A last dimension must equal K");

  const auto& b_shape = b->Shape();
  ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 2, "MatMulPacked weight B must be 2D [N, row_bytes]");
  ORT_RETURN_IF_NOT(b_shape[0] == n_, "MatMulPacked B dim 0 (", b_shape[0], ") != N (", n_, ")");
  const int64_t expected_rb = gguf_quant::GgmlRowBytes(static_cast<int>(ggml_type_), k_);
  ORT_RETURN_IF_NOT(expected_rb > 0 && b_shape[1] == expected_rb,
                    "MatMulPacked weight B row_bytes=", b_shape[1], " != expected=", expected_rb);

  MatMulPackedParams params{};
  params.b_data = b->Data<uint8_t>();
  params.global_scale = global_scale ? global_scale->Data<float>() : nullptr;
  params.dequantize_to_activation_dtype = dequantize_to_activation_dtype_;
  params.k = static_cast<int>(k_);
  params.n = static_cast<int>(n_);
  params.row_bytes = expected_rb;
  params.ggml_type = static_cast<int>(ggml_type_);

  TensorShapeVector y_dims = a_shape.AsShapeVector();
  y_dims[last_axis] = n_;
  Tensor* y = context->Output(0, TensorShape(y_dims));

  const int64_t m = a_shape.SizeToDimension(last_axis);
  if (m == 0) return Status::OK();

  constexpr bool is_bf16 = std::is_same_v<T, BFloat16>;
  ORT_RETURN_IF_NOT(!is_bf16 || GetDeviceProp().major >= 8,
                    "MatMulPacked BF16 requires compute capability 8.0 or higher");
  IAllocatorUniquePtr<T> scratch_w;
  IAllocatorUniquePtr<float> scratch_y;
  if (m > 8) {
    scratch_w = GetScratchBuffer<T>(SafeInt<size_t>(n_) * k_, GetComputeStream(context));
    if (global_scale) {
      scratch_y = GetScratchBuffer<float>(SafeInt<size_t>(m) * n_, GetComputeStream(context));
    }
  }

  return LaunchMatMulPackedCuda(
      GetCublasHandle(context),
      Stream(context),
      is_bf16,
      a->DataRaw(),
      params,
      scratch_w ? scratch_w.get() : nullptr,
      scratch_y ? scratch_y.get() : nullptr,
      y->MutableDataRaw(),
      m);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

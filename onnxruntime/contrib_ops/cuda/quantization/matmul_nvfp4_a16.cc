// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_nvfp4_a16.h"
#include "contrib_ops/cuda/math/matmul_block_scaled_fp4.h"

#include <type_traits>

#include "core/common/safeint.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime::contrib::cuda {
using namespace onnxruntime::cuda;

ONNX_OPERATOR_KERNEL_EX(
    MatMulNVFP4A16, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, BFloat16>())
        .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>())
        .TypeConstraint("T2", BuildKernelDefConstraints<float>()),
    MatMulNVFP4A16);

template <typename T>
Status MatMulNVFP4A16::ComputeImpl(OpKernelContext* context) const {
  using CudaT = typename ToCudaType<T>::MappedType;
  constexpr bool is_bf16 = std::is_same_v<T, BFloat16>;
  const auto* a = context->Input<Tensor>(0);
  const auto* b = context->Input<Tensor>(1);
  const auto* scales = context->Input<Tensor>(2);
  const auto* global_scale = context->Input<Tensor>(3);
  const auto& a_shape = a->Shape();
  const auto& b_shape = b->Shape();
  ORT_RETURN_IF_NOT(a_shape.NumDimensions() >= 1, "A must have rank at least 1.");
  ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 2, "B must have shape [N, K/2].");
  const int64_t n = b_shape[0];
  const int64_t k = SafeInt<int64_t>(b_shape[1]) * 2;
  ORT_RETURN_IF_NOT(k % 16 == 0, "K must be divisible by 16.");
  ORT_RETURN_IF_NOT(a_shape[a_shape.NumDimensions() - 1] == k, "A's last dimension must equal K.");
  ORT_RETURN_IF_NOT(scales->Shape().NumDimensions() == 2 &&
                        scales->Shape()[0] == n && scales->Shape()[1] == k / 16,
                    "weight_scale must have shape [N, K/16].");
  ORT_RETURN_IF_NOT(global_scale->Shape().Size() == 1, "weight_scale_2 must contain one element.");
  TensorShapeVector dims(a_shape.GetDims().begin(), a_shape.GetDims().end() - 1);
  dims.push_back(n);
  auto* y = context->Output(0, TensorShape(dims));
  if (y->Shape().Size() == 0) {
    return Status::OK();
  }
  const int m_i = SafeInt<int>(a_shape.SizeToDimension(a_shape.NumDimensions() - 1));
  const int n_i = SafeInt<int>(n);
  const int k_i = SafeInt<int>(k);
  const int64_t count = y->Shape().Size();
  if (k == 0) {
    return LaunchNVFP4A16Epilogue(y->MutableDataRaw(), nullptr, global_scale->Data<float>(),
                                  count, is_bf16, Stream(context));
  }

  // Only the tensor-core sub-path preserves FP32 products. The generic scalar GEMV
  // multiplies A by E2M1 in the activation type before applying the local scale.
  const int gemv_max_m = MatMulBlockQuantizedFp4WeightGemvMaxM(k_i, GetDeviceProp());
  if (m_i <= gemv_max_m && CanUseMatMulBlockQuantizedFp4WeightMma(k_i, GetDeviceProp())) {
    return LaunchMatMulBlockQuantizedFp4WeightGemv(
        y->MutableDataRaw(), a->DataRaw(), b->DataRaw(), scales->DataRaw(),
        global_scale->Data<float>(), nullptr, m_i, n_i, k_i, 16, is_bf16,
        GetDeviceProp(), Stream(context));
  }

  // Other shapes and disabled tensor-core GEMV use GEMM. The temporary weight excludes
  // the global scale; keep GEMM's output FP32 until the separate global-scale epilogue.
  auto local_weight = GetScratchBuffer<CudaT>(SafeInt<size_t>(n) * k, GetComputeStream(context));
  auto accumulator = GetScratchBuffer<float>(SafeInt<size_t>(count), GetComputeStream(context));
  ORT_RETURN_IF_ERROR(LaunchDequantizeNVFP4Local(
      local_weight.get(), b->Data<uint8_t>(), scales->Data<uint8_t>(), n, k,
      is_bf16, Stream(context)));
  const float one = 1.0f;
  const float zero = 0.0f;
  constexpr auto input_type = is_bf16 ? CUDA_R_16BF : CUDA_R_16F;
  CUBLAS_RETURN_IF_ERROR(cublasGemmEx(
      GetCublasHandle(context), CUBLAS_OP_T, CUBLAS_OP_N,
      n_i, m_i, k_i, &one, local_weight.get(), input_type, k_i,
      a->DataRaw(), input_type, k_i, &zero, accumulator.get(), CUDA_R_32F, n_i,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  return LaunchNVFP4A16Epilogue(
      y->MutableDataRaw(), accumulator.get(), global_scale->Data<float>(),
      count, is_bf16, Stream(context));
}

Status MatMulNVFP4A16::ComputeInternal(OpKernelContext* context) const {
  return context->Input<Tensor>(0)->IsDataType<BFloat16>() ? ComputeImpl<BFloat16>(context)
                                                           : ComputeImpl<MLFloat16>(context);
}

}  // namespace onnxruntime::contrib::cuda

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_nbits_swiglu.h"

#include <limits>

#include "contrib_ops/cuda/quantization/dequantize_blockwise.cuh"
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime::contrib::cuda {
using namespace onnxruntime::cuda;

ONNX_OPERATOR_KERNEL_EX(
    MatMulNBitsSwiGLU, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("TWeight", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBitsSwiGLU);

MatMulNBitsSwiGLU::MatMulNBitsSwiGLU(const OpKernelInfo& info) : CudaKernel(info) {
  int64_t k, n;
  ORT_ENFORCE(info.GetAttr("K", &k).IsOK() && info.GetAttr("N", &n).IsOK(), "K and N are required");
  ORT_ENFORCE(k > 0 && k % 128 == 0 && k <= std::numeric_limits<int>::max() &&
                  n > 0 && n % 16 == 0 && n <= std::numeric_limits<int>::max(),
              "K must be a positive multiple of 128 and N a positive multiple of 16, both fitting int32");
  k_ = static_cast<int>(k);
  n_ = static_cast<int>(n);
}

Status MatMulNBitsSwiGLU::ComputeInternal(OpKernelContext* context) const {
  const auto* x = context->Input<Tensor>(0);
  const auto* weight = context->Input<Tensor>(1);
  const auto* scales = context->Input<Tensor>(2);
  const auto& shape = x->Shape();
  ORT_RETURN_IF_NOT(shape.NumDimensions() == 2 && shape[1] == k_,
                    "X must have shape [M,K]");
  ORT_RETURN_IF_NOT(weight->Shape() == TensorShape({n_, k_ / 128, 32}) &&
                        scales->Shape() == TensorShape({n_, k_ / 128}),
                    "weight must be [N,K/128,32] and scales [N,K/128]");
  const int m = SafeInt<int>(shape[0]);
  auto* output = context->Output(0, TensorShape({m, n_ / 2}));
  if (m == 0) return Status::OK();
  ORT_RETURN_IF_NOT(GetDeviceProp().major >= 6, "MatMulNBitsSwiGLU requires SM60 or newer");
  const auto stream = Stream(context);
  if (m <= 8) {
    return LaunchMatMulNBitsSwiGLU(stream, x->DataRaw(), weight->Data<uint8_t>(),
                                   scales->DataRaw(), output->MutableDataRaw(), m, n_, k_);
  }

  // Match MatMulNBits' large-M dequantize/GEMM path; only fuse its elementwise consumers.
  auto dequantized = GetScratchBuffer<MLFloat16>(SafeInt<size_t>(n_) * k_, GetComputeStream(context));
  auto projected = GetScratchBuffer<MLFloat16>(SafeInt<size_t>(m) * n_, GetComputeStream(context));
  ORT_RETURN_IF_ERROR(DequantizeNBits(
      2, reinterpret_cast<half*>(dequantized.get()), weight->Data<uint8_t>(),
      reinterpret_cast<const half*>(scales->DataRaw()), static_cast<const uint8_t*>(nullptr),
      nullptr, k_, n_, 128, stream));
  const half alpha = half(1.0f), zero = half(0.0f);
  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      GetCublasHandle(context), CUBLAS_OP_T, CUBLAS_OP_N, n_, m, k_, &alpha,
      reinterpret_cast<const half*>(dequantized.get()), k_,
      reinterpret_cast<const half*>(x->DataRaw()), k_, &zero,
      reinterpret_cast<half*>(projected.get()), n_, GetDeviceProp(), UseTF32()));
  return LaunchMatMulNBitsSwiGLUEpilogue(stream, projected.get(), output->MutableDataRaw(), m, n_);
}

}  // namespace onnxruntime::contrib::cuda

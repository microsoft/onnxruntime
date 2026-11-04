// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_packed_projection.h"
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"

#include <limits>

namespace onnxruntime::contrib::cuda {

ONNX_OPERATOR_KERNEL_EX(MatMulPackedNVFP4, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>())
                            .TypeConstraint("T2", BuildKernelDefConstraints<float>()),
                        MatMulPackedNVFP4);

ONNX_OPERATOR_KERNEL_EX(MatMulPackedQ8, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>())
                            .TypeConstraint("S", BuildKernelDefConstraints<MLFloat16, BFloat16>()),
                        MatMulPackedQ8);

ONNX_OPERATOR_KERNEL_EX(MatMulPackedIQ2XXS, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", BuildKernelDefConstraints<BFloat16>())
                            .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>())
                            .TypeConstraint("S", BuildKernelDefConstraints<MLFloat16, BFloat16>()),
                        MatMulPackedIQ2XXS);

template <PackedProjectionFormat Format>
Status MatMulPackedProjection<Format>::ComputeInternal(OpKernelContext* context) const {
  const auto* a = context->Input<Tensor>(0);
  const auto* b = context->Input<Tensor>(1);
  const auto* scales = context->Input<Tensor>(2);
  const auto& as = a->Shape();
  const auto& bs = b->Shape();
  const auto& ss = scales->Shape();
  ORT_RETURN_IF_NOT(GetDeviceProp().major >= 8, "Packed projections require CUDA SM80 or newer.");
  ORT_RETURN_IF_NOT(as.NumDimensions() >= 1, "A must have rank at least one.");
  const int64_t k = as[as.NumDimensions() - 1];
  const int64_t rows = as.SizeToDimension(as.NumDimensions() - 1);
  ORT_RETURN_IF_NOT(rows >= 1 && rows <= 8, "Flattened activation rows must be in [1,8].");
  ORT_RETURN_IF_NOT(k > 0 && k <= std::numeric_limits<int>::max() && k % 64 == 0,
                    "K must be positive, fit int32, and be divisible by 64.");
  constexpr bool fp4 = Format == PackedProjectionFormat::NVFP4;
  constexpr bool iq2 = Format == PackedProjectionFormat::IQ2XXS;
  ORT_RETURN_IF_NOT(bs.NumDimensions() == (fp4 ? 2 : 3), "Invalid packed weight rank.");
  const int64_t n = bs[0];
  ORT_RETURN_IF_NOT(n > 0 && n <= std::numeric_limits<int>::max() - 32, "Invalid N.");
  ORT_RETURN_IF_NOT(ss.NumDimensions() == 2 && ss[0] == n, "Invalid scale shape.");
  if constexpr (fp4) {
    ORT_RETURN_IF_NOT(k % 1024 == 0 && bs[1] == k / 2 && ss[1] == k / 16,
                      "NVFP4 needs B [N,K/2], scales [N,K/16], K divisible by 1024.");
    ORT_RETURN_IF_NOT(rows != 1 || k == 4096 || k == 8192, "NVFP4 row-one path requires K=4096 or K=8192.");
    ORT_RETURN_IF_NOT(context->Input<Tensor>(3)->Shape().Size() == 1,
                      "Global scale must contain exactly one FP32 element.");
  } else {
    ORT_RETURN_IF_NOT(bs[1] == k / 32 && bs[2] == (iq2 ? 8 : 32),
                      "B must have shape [N,K/32,8] for IQ2XXS or [N,K/32,32] for Q8.");
    ORT_RETURN_IF_NOT(ss[1] == k / (iq2 ? 256 : 32), "Invalid scale block count.");
    if constexpr (iq2) {
      ORT_RETURN_IF_NOT(k % 256 == 0, "IQ2XXS requires K divisible by 256.");
    } else {
      ORT_RETURN_IF_NOT(rows != 1 || scales->IsDataType<MLFloat16>() || k <= 6144,
                        "Q8 BF16-scale row-one path requires K <= 6144.");
    }
  }
  const auto packed_bytes = SafeInt<size_t>(n) * (fp4 ? k / 2 : k / 32 * (iq2 ? 8 : 32));
  const auto output_elements = SafeInt<size_t>(rows) * n;
  ORT_UNUSED_PARAMETER(packed_bytes);
  ORT_UNUSED_PARAMETER(output_elements);
  TensorShapeVector output_dims(as.GetDims().begin(), as.GetDims().end() - 1);
  output_dims.push_back(n);
  auto* y = context->Output(0, TensorShape(output_dims));
  CUDA_RETURN_IF_ERROR(LaunchPackedProjection(
      Format, a->DataRaw(), b->Data<uint8_t>(), scales->DataRaw(),
      fp4 ? context->Input<Tensor>(3)->Data<float>() : nullptr, scales->IsDataType<MLFloat16>(), static_cast<int>(rows),
      static_cast<int>(n), static_cast<int>(k), y->MutableDataRaw(), Stream(context)));
  return Status::OK();
}

template class MatMulPackedProjection<PackedProjectionFormat::NVFP4>;
template class MatMulPackedProjection<PackedProjectionFormat::Q8>;
template class MatMulPackedProjection<PackedProjectionFormat::IQ2XXS>;

}  // namespace onnxruntime::contrib::cuda

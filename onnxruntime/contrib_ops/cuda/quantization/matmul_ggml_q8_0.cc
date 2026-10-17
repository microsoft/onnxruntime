// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_ggml_q8_0.h"

#include "core/common/safeint.h"

namespace onnxruntime::contrib::cuda {

ONNX_OPERATOR_KERNEL_EX(
    MatMulGGMLQ8_0, kMSDomain, 1, kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<float, MLFloat16, BFloat16>())
        .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>()),
    MatMulGGMLQ8_0);

Status MatMulGGMLQ8_0::ComputeInternal(OpKernelContext* context) const {
  const auto* a = context->Input<Tensor>(0);
  const auto* b = context->Input<Tensor>(1);
  const auto* bias = context->Input<Tensor>(2);
  const auto& a_shape = a->Shape();
  const auto& b_shape = b->Shape();
  ORT_RETURN_IF_NOT(a_shape.NumDimensions() >= 1, "A must have rank at least 1.");
  ORT_RETURN_IF_NOT(b_shape.NumDimensions() == 3 && b_shape[2] == 34,
                    "B must have shape [N, K/32, 34].");
  const int64_t n = b_shape[0];
  const int64_t k_blocks = b_shape[1];
  const int64_t k = SafeInt<int64_t>(k_blocks) * 32;
  ORT_RETURN_IF_NOT(a_shape[a_shape.NumDimensions() - 1] == k,
                    "A's last dimension must equal 32 times B's block count.");
  if (bias != nullptr) {
    ORT_RETURN_IF_NOT(bias->Shape().NumDimensions() == 1 && bias->Shape()[0] == n,
                      "bias must have shape [N].");
  }
  TensorShapeVector output_dims(a_shape.GetDims().begin(), a_shape.GetDims().end() - 1);
  output_dims.push_back(n);
  auto* y = context->Output(0, TensorShape(output_dims));
  if (y->Shape().Size() == 0) {
    return Status::OK();
  }
  const int64_t m = a_shape.SizeToDimension(a_shape.NumDimensions() - 1);
  const auto packed_bytes = SafeInt<size_t>(n) * k_blocks * 34;
  const auto output_elements = SafeInt<int64_t>(m) * n;
  ORT_UNUSED_PARAMETER(packed_bytes);
  ORT_UNUSED_PARAMETER(output_elements);
  const auto type = a->IsDataType<float>()       ? Q8ActivationType::Float
                    : a->IsDataType<MLFloat16>() ? Q8ActivationType::Float16
                                                 : Q8ActivationType::BFloat16;
  return LaunchMatMulGGMLQ8_0(a->DataRaw(), b->Data<uint8_t>(),
                              bias != nullptr ? bias->DataRaw() : nullptr, y->MutableDataRaw(),
                              m, n, k_blocks, type, Stream(context));
}

}  // namespace onnxruntime::contrib::cuda

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gsl/span>
#include "mul.h"
#include "utils.h"

// Defines a kernel creation function for version 14 of Mul.
ONNX_OPERATOR_KERNEL_EX(
    Mul,
    kOnnxDomain,
    /*version*/ 14,  // Equivalent to start_version: 14, end_version: 14 (inclusive)
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Mul)

Mul::Mul(const OrtKernelInfo* info, void* state, PrivateTag) : BaseKernelImpl(info, state) {}

/*static*/
OrtStatus* Mul::Create(const OrtKernelInfo* info, void* state,
                       /*out*/ std::unique_ptr<Mul>& result) {
  // Note: can do basic validation or preprocessing via the OrtKernelInfo APIs.
  result = std::make_unique<Mul>(info, state, PrivateTag{});
  return nullptr;
}

OrtStatus* Mul::DoCompute(OrtKernelContext* kernel_ctx) {
  Ort::KernelContext kernel_context(kernel_ctx);
  static_cast<void>(this->state_);  // NOTE: Unused in this example.
  static_cast<void>(this->info_);   // NOTE: Unused in this example.

  gsl::span<const float> input0;
  gsl::span<const float> input1;
  std::vector<int64_t> shape0;
  std::vector<int64_t> shape1;

  RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 0, input0, shape0));
  RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 1, input1, shape1));
  RETURN_IF(shape0 != shape1, Ort::GetApi(), "Mul kernel doesn't support broadcasting.");  // Checked by GetCapability

  Ort::UnownedValue output = kernel_context.GetOutput(0, shape0);
  float* output_data = output.GetTensorMutableData<float>();

  for (size_t i = 0; i < input0.size(); ++i) {
    output_data[i] = input0[i] * input1[i];
  }

  return nullptr;
}

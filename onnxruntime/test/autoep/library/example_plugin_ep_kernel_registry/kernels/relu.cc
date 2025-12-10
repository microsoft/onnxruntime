// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "relu.h"

#include <gsl/span>
#include <algorithm>
#include <cassert>

#include "utils.h"

// Defines a kernel creation function for version 14 of Relu.
ONNX_OPERATOR_KERNEL_EX(
    Relu,
    kOnnxDomain,
    /*version*/ 14,  // Equivalent to start_version: 14, end_version: 14 (inclusive)
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddInputOutputMutableAlias(0, 0)),
    Relu)

Relu::Relu(const OrtKernelInfo* info, void* state, PrivateTag) : BaseKernelImpl(info, state) {}

/*static*/
OrtStatus* Relu::Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Relu>& kernel) {
  Ort::ConstKernelInfo kernel_info(info);
  kernel = std::make_unique<Relu>(info, state, PrivateTag{});
  return nullptr;
}

OrtStatus* Relu::DoCompute(OrtKernelContext* kernel_ctx) {
  Ort::KernelContext kernel_context(kernel_ctx);
  static_cast<void>(this->state_);  // NOTE: Unused in this example.
  static_cast<void>(this->info_);   // NOTE: Unused in this example.

  gsl::span<const float> input0;
  std::vector<int64_t> shape0;
  RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 0, input0, shape0));

  Ort::UnownedValue output = kernel_context.GetOutput(0, shape0);
  float* output_data = output.GetTensorMutableData<float>();

  for (size_t i = 0; i < input0.size(); ++i) {
    output_data[i] = std::max(0.0f, input0[i]);
  }

  return nullptr;
}

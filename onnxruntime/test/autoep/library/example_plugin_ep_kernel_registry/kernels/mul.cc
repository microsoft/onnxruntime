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
  static_cast<void>(this->data_transfer_impl_);  // NOTE: Unused in this example.
  static_cast<void>(this->info_);                // NOTE: Unused in this example.

  // Get first input's data.
  gsl::span<const float> input0;
  std::vector<int64_t> shape0;
  RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 0, input0, shape0));

  // Get second input's data.
  // This second input may have been pre-packed if it is a constant weight.
  Ort::ConstValue ort_value1 = packed_weight_1_.has_value() ? packed_weight_1_->GetConst() : kernel_context.GetInput(1);
  gsl::span<const float> input1;
  std::vector<int64_t> shape1;
  RETURN_IF_ERROR(GetValueDataAndShape<float>(ort_value1, input1, shape1));

  RETURN_IF(shape0 != shape1, Ort::GetApi(), "Mul kernel doesn't support broadcasting.");  // Checked by GetCapability

  Ort::UnownedValue output = kernel_context.GetOutput(0, shape0);
  float* output_data = output.GetTensorMutableData<float>();

  for (size_t i = 0; i < input0.size(); ++i) {
    output_data[i] = input0[i] * input1[i];
  }

  return nullptr;
}

OrtStatus* Mul::DoPrePackConstantTensor(const OrtValue* tensor, int input_index, OrtAllocator* alloc,
                                        /*out*/ bool& is_packed) {
  // This example Mul kernel does not really need to pre-pack mul initializers, but we show it here as an example.
  // This implementation just copies original tensor without modification. An actual implementation would, for example,
  // transform to an appropriate data layout.

  if (input_index != 1) {
    is_packed = false;
    return nullptr;
  }

  Ort::ConstValue original_weight(tensor);
  auto type_shape_info = original_weight.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = type_shape_info.GetShape();
  auto elem_type = type_shape_info.GetElementType();

  packed_weight_1_ = Ort::Value::CreateTensor(alloc, shape.data(), shape.size(), elem_type);
  RETURN_IF_ERROR(CopyTensor(original_weight, packed_weight_1_->GetUnowned()));
  is_packed = true;

  return nullptr;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gsl/span>
#include "mul.h"
#include "utils.h"

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Mul,
    kOnnxDomain,
    7, 24,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", MLDataTypes::GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Mul)

Mul::Mul(const OrtKernelInfo* info, void* state, PrivateTag) : info_{info}, state_{state} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* Mul::Create(const OrtKernelInfo* info, void* state,
                       /*out*/ std::unique_ptr<Mul>& result) {
  // Note: can do basic validation or preprocessing via the OrtKernelInfo APIs.
  result = std::make_unique<Mul>(info, state, PrivateTag{});
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL Mul::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  Mul* mul = static_cast<Mul*>(this_ptr);
  return mul->DoCompute(kernel_ctx);
}

/*static*/
void ORT_API_CALL Mul::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Mul*>(this_ptr);
}

OrtStatus* Mul::DoCompute(OrtKernelContext* kernel_ctx) noexcept {
  Ort::KernelContext kernel_context(kernel_ctx);
  (void)this->state_;  // NOTE: Unused in this example.
  (void)this->info_;   // NOTE: Unused in this example.

  try {
    gsl::span<const float> input0;
    gsl::span<const float> input1;
    std::vector<int64_t> shape0;
    std::vector<int64_t> shape1;

    RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 0, input0, shape0));
    RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 1, input1, shape1));

    if (shape0 != shape1) {
      Ort::Status status("Mul kernel does not support broadcasting", ORT_EP_FAIL);
      return status.release();
    }

    Ort::UnownedValue output = kernel_context.GetOutput(0, shape0);
    float* output_data = output.GetTensorMutableData<float>();

    for (size_t i = 0; i < input0.size(); ++i) {
      output_data[i] = input0[i] * input1[i];
    }
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

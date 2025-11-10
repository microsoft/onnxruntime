// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "relu.h"

#include <gsl/span>
#include <algorithm>
#include <cassert>

#include "utils.h"

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Relu,
    kOnnxDomain,
    14, 24,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", MLDataTypes::GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddInputOutputMutableAlias({0, 0})),
    Relu)

Relu::Relu(const OrtKernelInfo* info, void* state, PrivateTag)
    : info_(info),
      state_(state) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* Relu::Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Relu>& kernel) {
  Ort::ConstKernelInfo kernel_info(info);

  try {
    kernel = std::make_unique<Relu>(info, state, PrivateTag{});
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL Relu::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  Relu* relu = static_cast<Relu*>(this_ptr);
  return relu->DoCompute(kernel_ctx);
}

/*static*/
void ORT_API_CALL Relu::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Relu*>(this_ptr);
}

OrtStatus* Relu::DoCompute(OrtKernelContext* kernel_ctx) noexcept {
  Ort::KernelContext kernel_context(kernel_ctx);
  (void)this->state_;  // NOTE: Unused in this example.
  (void)this->info_;   // NOTE: Unused in this example.

  try {
    gsl::span<const float> input0;
    std::vector<int64_t> shape0;
    RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 0, input0, shape0));

    Ort::UnownedValue output = kernel_context.GetOutput(0, shape0);
    float* output_data = output.GetTensorMutableData<float>();

    for (size_t i = 0; i < input0.size(); ++i) {
      output_data[i] = std::max(0.0f, input0[i]);
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

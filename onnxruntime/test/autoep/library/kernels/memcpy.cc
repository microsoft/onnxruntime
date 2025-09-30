// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "memcpy.h"
#include "utils.h"

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemType::OrtMemTypeCPUInput)
         .AddTypeConstraint("T", MLDataTypes::GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Memcpy)

Memcpy::Memcpy(ApiPtrs api, const OrtKernelInfo* info) : api{api}, info{info} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}
/*static*/
OrtStatus* Memcpy::Create(const OrtKernelInfo* info,
                          /*out*/ std::unique_ptr<Memcpy>& result) {
  ApiPtrs api = {Ort::GetApi(), Ort::GetEpApi(), Ort::GetModelEditorApi()};
  result = std::make_unique<Memcpy>(api, info);
  return nullptr;
}

/*static*/
OrtStatus* ORT_API_CALL Memcpy::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  Memcpy* memcpy = static_cast<Memcpy*>(this_ptr);
  return memcpy->DoCompute(kernel_ctx);
}

/*static*/
void ORT_API_CALL Memcpy::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Memcpy*>(this_ptr);
}

OrtStatus* Memcpy::DoCompute(OrtKernelContext* kernel_ctx) noexcept {
  // TODO: Use DataTransfer
  const OrtApi& ort_api = api.ort_api;
  Ort::KernelContext kernel_context(kernel_ctx);
  try {
    size_t num_inputs = kernel_context.GetInputCount();
    RETURN_IF(num_inputs != 1, ort_api, "Expected only 1 input for MemcpyFromHost kernel");

    gsl::span<const float> input0;
    std::vector<int64_t> shape0;
    GetKernelInputDataAndShape(kernel_context, 0, input0, shape0);

    size_t num_outputs = kernel_context.GetOutputCount();
    RETURN_IF(num_outputs != 1, ort_api, "Expected only 1 output for MemcpyFromHost kernel");

    auto output = kernel_context.GetOutput(0, shape0);
    float* output_data = output.GetTensorMutableData<float>();

    for (size_t i = 0; i < input0.size(); ++i) {
      output_data[i] = input0[i];  // straight copy
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

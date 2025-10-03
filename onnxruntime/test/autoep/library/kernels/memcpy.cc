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
         // .AddTypeConstraint("T", MLDataTypes::GetAllFixedSizeTensorTypesIRv9()),
         .AddTypeConstraint("T", MLDataTypes::GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Memcpy)

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    (Ort::KernelDefBuilder()
         .SetOutputMemType(0, OrtMemType::OrtMemTypeCPUOutput)
         // .AddTypeConstraint("T", MLDataTypes::GetAllFixedSizeTensorTypesIRv9()),
         .AddTypeConstraint("T", MLDataTypes::GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Memcpy)

Memcpy::Memcpy(const OrtKernelInfo* info, void* state) : info_{info}, state_{state} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* Memcpy::Create(const OrtKernelInfo* info, void* state,
                          /*out*/ std::unique_ptr<Memcpy>& result) {
  const OrtApi& ort_api = Ort::GetApi();

  try {
    Ort::ConstKernelInfo kernel_info(info);

    // Basic validation before creating kernel.
    size_t num_inputs = kernel_info.GetInputCount();
    size_t num_outputs = kernel_info.GetOutputCount();
    RETURN_IF(num_inputs != 1, ort_api, "Expected only 1 input for Memcpy kernel");
    RETURN_IF(num_outputs != 1, ort_api, "Expected only 1 output for Memcpy kernel");

    result = std::make_unique<Memcpy>(info, state);
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
OrtStatus* ORT_API_CALL Memcpy::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  Memcpy* memcpy = static_cast<Memcpy*>(this_ptr);
  return memcpy->DoCompute(kernel_ctx);
}

/*static*/
void ORT_API_CALL Memcpy::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Memcpy*>(this_ptr);
}

OrtStatus* Memcpy::DoCompute(OrtKernelContext* kernel_ctx) noexcept {
  const OrtEpApi& ep_api = Ort::GetEpApi();
  Ort::KernelContext kernel_context(kernel_ctx);

  try {
    Ort::ConstValue input = kernel_context.GetInput(0);
    std::vector<int64_t> shape = input.GetTensorTypeAndShapeInfo().GetShape();
    Ort::UnownedValue output = kernel_context.GetOutput(0, shape);

    std::array<const OrtValue*, 1> src_tensors = {input};
    std::array<OrtValue*, 1> dst_tensors = {output};

    RETURN_IF_ERROR(ep_api.KernelInfo_CopyTensors(info_,
                                                  src_tensors.data(),
                                                  dst_tensors.data(),
                                                  /*stream*/ nullptr,
                                                  src_tensors.size()));
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

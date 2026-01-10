// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "loop.h"

#include <gsl/gsl>
#include "utils.h"
#include "../ep.h"

// Defines a kernel creation function for Loop opset 21
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Loop,
    kOnnxDomain,
    /*start version*/ 21, /*end version*/ 22,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'M' needs to be on CPU
         .SetInputMemType(1, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    LoopHelper)

// Defines a kernel creation function for Loop opset 23
ONNX_OPERATOR_KERNEL_EX(
    Loop,
    kOnnxDomain,
    /*version*/ 23,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'M' needs to be on CPU
         .SetInputMemType(1, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    LoopHelper)

// Defines a kernel creation function for Loop opset 24
ONNX_OPERATOR_KERNEL_EX(
    Loop,
    kOnnxDomain,
    /*version*/ 24,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'M' needs to be on CPU
         .SetInputMemType(1, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    LoopHelper)

/*static*/
OrtStatus* LoopHelper::CreateKernelImpl(const OrtKernelInfo* ort_kernel_info, void* state,
                                        /*out*/ OrtKernelImpl*& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  const OrtEpApi& ep_api = Ort::GetEpApi();
  Ort::ConstKernelInfo kernel_info(ort_kernel_info);

  // Ask ORT to create a OrtKernelImpl for Loop.
  auto loop_helper = std::make_unique<LoopHelper>(kernel_info, state);
  RETURN_IF_ERROR(ep_api.CreateLoopKernel(kernel_info, loop_helper.get(), &kernel));
  loop_helper.release();  // ORT owns this instance on successful call to CreateLoopKernel.

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

LoopHelper::LoopHelper(Ort::ConstKernelInfo info, void* state)
    : OrtLoopKernelHelper{},  // Initialize all OrtLoopKernelHelper members to NULL/zero
      info_{info},
      data_transfer_impl_{reinterpret_cast<OrtDataTransferImpl*>(state)} {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  ConcatOutput = ConcatOutputImpl;
}

/*static*/
void ORT_API_CALL LoopHelper::ReleaseImpl(_In_ OrtLoopKernelHelper* this_ptr) noexcept {
  delete static_cast<LoopHelper*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL LoopHelper::ConcatOutputImpl(
    _In_ OrtLoopKernelHelper* this_ptr,
    _In_opt_ void* /*stream_handle*/,
    _In_reads_(num_per_iteration_outputs) const OrtValue* const* per_iteration_outputs,
    _In_ size_t num_per_iteration_outputs,
    _Out_writes_bytes_all_(output_size_in_bytes) void* output,
    _In_ size_t output_size_in_bytes) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // Concatenates loop iteration outputs. Ignores native stream handle argument as this example EP kernel
  // uses CPU memory. Based on the default implementation in CPU EP.

  // An actual implementation can retrieve state from the OrtLoopKernelHelper (e.g., OrtDataTransferImpl, etc.).
  LoopHelper* loop_kernel_helper = static_cast<LoopHelper*>(this_ptr);
  (void)loop_kernel_helper->info_;                // Unused in this example.
  (void)loop_kernel_helper->data_transfer_impl_;  // Unused in this example.

  Ort::ConstValue first_output{per_iteration_outputs[0]};
  Ort::TensorTypeAndShapeInfo type_shape = first_output.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> per_iteration_shape = type_shape.GetShape();
  size_t bytes_per_iteration = first_output.GetTensorSizeInBytes();

  gsl::span<std::byte> output_span = gsl::make_span<std::byte>(static_cast<std::byte*>(output),
                                                               output_size_in_bytes);

  for (size_t i = 0; i < num_per_iteration_outputs; i++) {
    Ort::ConstValue ort_value{per_iteration_outputs[i]};

    // Sanity check that all OrtValue's have the same amount of data.
    RETURN_IF(bytes_per_iteration != ort_value.GetTensorSizeInBytes(), Ort::GetApi(),
              "OrtLoopConcatOutputFunc received outputs with different sizes.");

    auto src = gsl::make_span<const std::byte>(static_cast<const std::byte*>(ort_value.GetTensorRawData()),
                                               bytes_per_iteration);
    auto dst = output_span.subspan(i * bytes_per_iteration, bytes_per_iteration);
    gsl::copy(src, dst);
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

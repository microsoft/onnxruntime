// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "loop.h"

#include <gsl/gsl>
#include "utils.h"

// Defines a kernel creation function for Loop opset 13
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Loop,
    kOnnxDomain,
    /*start version*/ 13, /*end version*/ 15,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'M' needs to be on CPU
         .SetInputMemType(1, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Loop)

// Defines a kernel creation function for Loop opset 16
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Loop,
    kOnnxDomain,
    /*start version*/ 16, /*end version*/ 18,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'M' needs to be on CPU
         .SetInputMemType(1, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Loop)

// Defines a kernel creation function for Loop opset 19
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Loop,
    kOnnxDomain,
    /*start version*/ 19, /*end version*/ 20,
    (Ort::KernelDefBuilder()
         .SetInputMemType(0, OrtMemTypeCPUInput)  // 'M' needs to be on CPU
         .SetInputMemType(1, OrtMemTypeCPUInput)  // 'cond' needs to be on CPU
         .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("B", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Loop)

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
    Loop)

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
    Loop)

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
    Loop)

// Concatenates loop iteration outputs. Ignores native stream handle argument as this example EP kernel
// uses CPU memory.
// Based on the default implementation in CPU EP.
static OrtStatus* ORT_API_CALL ConcatLoopOutput(void* state,
                                                void* /*stream_handle*/,
                                                OrtValue* const* per_iteration_output,
                                                size_t num_iteration_outputs,
                                                void* output,
                                                size_t output_size_in_bytes) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // An actual implementation can retrieve state from the Loop OrtKernelImpl (e.g., OrtDataTransferImpl, etc.).
  Loop* loop_kernel = reinterpret_cast<Loop*>(state);
  (void)loop_kernel;  // Note: Unused in this example.

  Ort::ConstValue first_output{per_iteration_output[0]};
  Ort::TensorTypeAndShapeInfo type_shape = first_output.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> per_iteration_shape = type_shape.GetShape();
  size_t bytes_per_iteration = first_output.GetTensorSizeInBytes();

  gsl::span<std::byte> output_span = gsl::make_span<std::byte>(static_cast<std::byte*>(output),
                                                               output_size_in_bytes);

  for (size_t i = 0; i < num_iteration_outputs; i++) {
    Ort::ConstValue ort_value{per_iteration_output[i]};

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

/*static*/
OrtStatus* Loop::Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Loop>& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  const OrtEpApi& ep_api = Ort::GetEpApi();
  auto kernel_unique_ptr = std::make_unique<Loop>(info, state, PrivateTag{});

  // Create configuration with helper to concatenate loop outputs.
  OrtLoopKernelConfig* config = nullptr;
  RETURN_IF_ERROR(ep_api.CreateLoopKernelConfig(&config));
  Ort::Status status{ep_api.LoopKernelConfig_SetConcatOutputFunc(config, ConcatLoopOutput, kernel_unique_ptr.get())};

  if (!status.IsOK()) {
    ep_api.ReleaseLoopKernelConfig(config);  // TODO: Add CXX API for RAII
    return status.release();
  }

  // Create actual Loop kernel implementation.
  Ort::Status status2{ep_api.CreateLoopKernel(info, config, &kernel_unique_ptr->control_flow_kernel_)};

  if (!status2.IsOK()) {
    ep_api.ReleaseLoopKernelConfig(config);  // TODO: Add CXX API for RAII
    return status2.release();
  }

  kernel = std::move(kernel_unique_ptr);

  ep_api.ReleaseLoopKernelConfig(config);  // TODO: Add CXX API for RAII
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

Loop::Loop(const OrtKernelInfo* info, void* state, PrivateTag)
    : OrtKernelImpl{},  // Initialize all OrtKernelImpl functions to NULL
      info_{info},
      data_transfer_impl_{reinterpret_cast<OrtDataTransferImpl*>(state)},
      control_flow_kernel_{nullptr} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
  GetControlFlowKernel = GetControlFlowKernelImpl;
}

Loop::~Loop() {
  Ort::GetEpApi().ReleaseKernelImpl(control_flow_kernel_);
}

/*static*/
OrtStatus* ORT_API_CALL Loop::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Loop* loop_kernel = static_cast<Loop*>(this_ptr);
  static_cast<void>(loop_kernel->info_);                // NOTE: Unused in this example.
  static_cast<void>(loop_kernel->data_transfer_impl_);  // NOTE: Unused in this example.

  return loop_kernel->control_flow_kernel_->Compute(loop_kernel->control_flow_kernel_, kernel_ctx);
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
void ORT_API_CALL Loop::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Loop*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL Loop::GetControlFlowKernelImpl(OrtKernelImpl* this_ptr, OrtKernelImpl** out) noexcept {
  Loop* loop_kernel = static_cast<Loop*>(this_ptr);
  *out = loop_kernel->control_flow_kernel_;
  return nullptr;
}

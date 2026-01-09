// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "scan.h"

#include <gsl/gsl>
#include "utils.h"

// Defines a kernel creation function for Scan opset 21
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Scan,
    kOnnxDomain,
    /*start version*/ 21, /*end version*/ 22,
    (Ort::KernelDefBuilder()
         // 'I' is in the ONNX spec but is not used for any inputs or outputs
         // .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Scan)

// Defines a kernel creation function for Scan opset 23
ONNX_OPERATOR_KERNEL_EX(
    Scan,
    kOnnxDomain,
    /*version*/ 23,
    (Ort::KernelDefBuilder()
         // 'I' is in the ONNX spec but is not used for any inputs or outputs
         // .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Scan)

// Defines a kernel creation function for Scan opset 24
ONNX_OPERATOR_KERNEL_EX(
    Scan,
    kOnnxDomain,
    /*version*/ 24,
    (Ort::KernelDefBuilder()
         // 'I' is in the ONNX spec but is not used for any inputs or outputs
         // .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    Scan)

static OrtStatus* ORT_API_CALL Transpose(void* state, const size_t* permutation, size_t num_perm_elems,
                                         const OrtValue* ort_input, OrtSyncStream* stream,
                                         OrtValue* ort_output) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // An actual implementation can retrieve state from the Scan OrtKernelImpl (e.g., OrtDataTransferImpl, etc.).
  Scan* scan_kernel = reinterpret_cast<Scan*>(state);
  (void)scan_kernel;  // Note: Unused in this example.
  (void)stream;

  Ort::ConstValue input(ort_input);
  Ort::UnownedValue output(ort_output);
  gsl::span<const size_t> perm(permutation, num_perm_elems);

  // Note: This example implementation only supports 2D transpose (perm: [1, 0]) for convenience. A correct implementation
  // should support more general dimensions/permutations.
  RETURN_IF(perm.size() != 2 || perm[0] != 1 || perm[1] != 0, Ort::GetApi(),
            "Scan kernel for ExampleKernelEp only supports 2D transpose.");

  Ort::TensorTypeAndShapeInfo input_type_shape = input.GetTensorTypeAndShapeInfo();
  Ort::TensorTypeAndShapeInfo output_type_shape = output.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> input_shape = input_type_shape.GetShape();
  size_t num_elems = input_type_shape.GetElementCount();

  RETURN_IF(output_type_shape.GetElementCount() != num_elems, Ort::GetApi(),
            "Expected input and output of Scan's transpose helper to have the same number of elements");

  gsl::span<const float> src(input.GetTensorData<float>(), num_elems);
  gsl::span<float> dst(output.GetTensorMutableData<float>(), num_elems);

  size_t num_rows = static_cast<size_t>(input_shape[0]);
  size_t num_cols = static_cast<size_t>(input_shape[1]);

  for (size_t r = 0; r < num_rows; r++) {
    for (size_t c = 0; c < num_cols; c++) {
      size_t src_idx = r * num_cols + c;
      size_t dst_idx = c * num_rows + r;
      dst[dst_idx] = src[src_idx];
    }
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
OrtStatus* Scan::Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Scan>& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  const OrtEpApi& ep_api = Ort::GetEpApi();
  auto kernel_unique_ptr = std::make_unique<Scan>(info, state, PrivateTag{});

  // Ask ORT to create a OrtKernelImpl for Scan. The EP author provides a helper function
  // for transposing tensors. We pass `this` as the transpose function state, which
  // allows retrieval of EP resources (e.g., allocators, data transfer, etc.).
  void* transpose_func_state = kernel_unique_ptr.get();
  RETURN_IF_ERROR(ep_api.CreateScanKernel(info, Transpose, transpose_func_state,
                                          &kernel_unique_ptr->control_flow_kernel_));

  kernel = std::move(kernel_unique_ptr);
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

Scan::Scan(const OrtKernelInfo* info, void* state, PrivateTag)
    : OrtKernelImpl{},  // Initialize all OrtKernelImpl functions to NULL
      info_{info},
      data_transfer_impl_{reinterpret_cast<OrtDataTransferImpl*>(state)},
      control_flow_kernel_{nullptr} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
  GetControlFlowKernel = GetControlFlowKernelImpl;
}

Scan::~Scan() {
  Ort::GetEpApi().ReleaseKernelImpl(control_flow_kernel_);
}

/*static*/
OrtStatus* ORT_API_CALL Scan::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Scan* scan_kernel = static_cast<Scan*>(this_ptr);
  static_cast<void>(scan_kernel->info_);                // NOTE: Unused in this example.
  static_cast<void>(scan_kernel->data_transfer_impl_);  // NOTE: Unused in this example.

  return scan_kernel->control_flow_kernel_->Compute(scan_kernel->control_flow_kernel_, kernel_ctx);
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
void ORT_API_CALL Scan::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Scan*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL Scan::GetControlFlowKernelImpl(OrtKernelImpl* this_ptr, OrtKernelImpl** out) noexcept {
  Scan* scan_kernel = static_cast<Scan*>(this_ptr);
  *out = scan_kernel->control_flow_kernel_;
  return nullptr;
}

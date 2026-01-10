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
    ScanHelper)

// Defines a kernel creation function for Scan opset 23
ONNX_OPERATOR_KERNEL_EX(
    Scan,
    kOnnxDomain,
    /*version*/ 23,
    (Ort::KernelDefBuilder()
         // 'I' is in the ONNX spec but is not used for any inputs or outputs
         // .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    ScanHelper)

// Defines a kernel creation function for Scan opset 24
ONNX_OPERATOR_KERNEL_EX(
    Scan,
    kOnnxDomain,
    /*version*/ 24,
    (Ort::KernelDefBuilder()
         // 'I' is in the ONNX spec but is not used for any inputs or outputs
         // .AddTypeConstraint("I", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddTypeConstraint("V", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))),
    ScanHelper)

/*static*/
OrtStatus* ScanHelper::CreateKernelImpl(const OrtKernelInfo* ort_kernel_info, void* state,
                                        /*out*/ OrtKernelImpl*& kernel) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  const OrtEpApi& ep_api = Ort::GetEpApi();
  Ort::ConstKernelInfo kernel_info(ort_kernel_info);

  // Ask ORT to create a OrtKernelImpl for Scan.
  auto scan_helper = std::make_unique<ScanHelper>(kernel_info, state);
  RETURN_IF_ERROR(ep_api.CreateScanKernel(kernel_info, scan_helper.get(), &kernel));
  scan_helper.release();  // ORT owns this instance on successful call to CreateScanKernel.

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

ScanHelper::ScanHelper(Ort::ConstKernelInfo info, void* state)
    : OrtScanKernelHelper{},  // Initialize all OrtScanKernelHelper members to NULL/zero
      info_{info},
      data_transfer_impl_{reinterpret_cast<OrtDataTransferImpl*>(state)} {
  ort_version_supported = ORT_API_VERSION;
  Release = ReleaseImpl;
  Transpose = TransposeImpl;
}

/*static*/
void ORT_API_CALL ScanHelper::ReleaseImpl(_In_ OrtScanKernelHelper* this_ptr) noexcept {
  delete static_cast<ScanHelper*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL ScanHelper::TransposeImpl(_In_ OrtScanKernelHelper* this_ptr,
                                                  _In_reads_(num_permutation_elems) const size_t* permutation,
                                                  _In_ size_t num_permutation_elems,
                                                  _In_ const OrtValue* ort_input, _In_opt_ OrtSyncStream* /*stream*/,
                                                  _Inout_ OrtValue* ort_output) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // An actual implementation can retrieve state from the OrtScanKernelHelper (e.g., OrtDataTransferImpl, etc.).
  ScanHelper* scan_kernel_helper = static_cast<ScanHelper*>(this_ptr);
  (void)scan_kernel_helper->info_;                // Unused in this example.
  (void)scan_kernel_helper->data_transfer_impl_;  // Unused in this example.

  Ort::ConstValue input(ort_input);
  Ort::UnownedValue output(ort_output);
  gsl::span<const size_t> perm(permutation, num_permutation_elems);

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

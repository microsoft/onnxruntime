// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gsl/span>
#include <sstream>
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

Mul::Mul(const OrtKernelInfo* info, void* state, PrivateTag)
    : OrtKernelImpl{},  // Initialize all OrtKernelImpl functions to NULL
      info_{info},
      data_transfer_impl_{reinterpret_cast<OrtDataTransferImpl*>(state)} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;

  // Optional functions that are only needed to pre-pack weights. This Mul kernel pre-packs
  // input[1] weights as an example (not typically done by an actual implementation of Mul).
  PrePackWeight = PrePackWeightImpl;
  SetSharedPrePackedWeight = SetSharedPrePackedWeightImpl;
}

/*static*/
OrtStatus* Mul::Create(const OrtKernelInfo* info, void* state,
                       /*out*/ std::unique_ptr<Mul>& result) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  // Note: can do basic validation or preprocessing via the OrtKernelInfo APIs.
  result = std::make_unique<Mul>(info, state, PrivateTag{});
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
void ORT_API_CALL Mul::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Mul*>(this_ptr);
}

/*static*/
OrtStatus* ORT_API_CALL Mul::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Mul* mul_kernel = static_cast<Mul*>(this_ptr);
  static_cast<void>(mul_kernel->info_);  // NOTE: Unused in this example.

  Ort::KernelContext kernel_context(kernel_ctx);

  // Get first input's data.
  gsl::span<const float> input0;
  std::vector<int64_t> shape0;
  RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 0, input0, shape0));

  // Get second input's data.
  // This second input may have been pre-packed if it is a constant weight.
  gsl::span<const float> input1;
  std::vector<int64_t> shape1;

  if (mul_kernel->packed_weight_1_info_.has_value()) {
    const PackedWeightInfo& packed_weight_info = *mul_kernel->packed_weight_1_info_;
    shape1 = packed_weight_info.shape;

    size_t num_elems = 1;
    for (auto s : shape1) {
      num_elems *= s;
    }

    const float* input1_data = packed_weight_info.shared_data != nullptr
                                   ? reinterpret_cast<const float*>(packed_weight_info.shared_data)
                                   : reinterpret_cast<const float*>(packed_weight_info.owned_data.get());

    input1 = gsl::span<const float>(input1_data, num_elems);
  } else {
    RETURN_IF_ERROR(GetValueDataAndShape<float>(kernel_context.GetInput(1), input1, shape1));
  }

  RETURN_IF(shape0 != shape1, Ort::GetApi(), "Mul kernel doesn't support broadcasting.");  // Checked by GetCapability

  Ort::UnownedValue output = kernel_context.GetOutput(0, shape0);
  float* output_data = output.GetTensorMutableData<float>();

  for (size_t i = 0; i < input0.size(); ++i) {
    output_data[i] = input0[i] * input1[i];
  }

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL Mul::PrePackWeightImpl(OrtKernelImpl* this_ptr, const OrtValue* tensor,
                                               int input_index, OrtAllocator* allocator,
                                               OrtSharedPrePackedWeightCache* prepacked_weight_cache,
                                               /*out*/ bool* is_packed) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Mul* mul_kernel = static_cast<Mul*>(this_ptr);

  // This example Mul kernel does not really need to pre-pack mul initializers, but we show it here as an example.
  // This implementation just copies original tensor without modification. An actual implementation would, for example,
  // transform to an appropriate data layout.

  if (input_index != 1) {
    *is_packed = false;
    return nullptr;
  }

  Ort::ConstValue original_weight(tensor);
  auto type_shape_info = original_weight.GetTensorTypeAndShapeInfo();
  size_t num_bytes = original_weight.GetTensorSizeInBytes();

  PackedWeightInfo weight_info = {};
  weight_info.mem_info = Ort::ConstMemoryInfo(allocator->Info(allocator));
  weight_info.shape = type_shape_info.GetShape();
  weight_info.elem_type = type_shape_info.GetElementType();
  weight_info.num_bytes = num_bytes;
  weight_info.owned_data = AllocateBytes(allocator, num_bytes);

  // Note: This Ort::Value does not own the underlying data.
  Ort::Value packed_weight = Ort::Value::CreateTensor(weight_info.mem_info, weight_info.owned_data.get(),
                                                      weight_info.num_bytes, weight_info.shape.data(),
                                                      weight_info.shape.size(), weight_info.elem_type);

  RETURN_IF_ERROR(CopyTensor(*mul_kernel->data_transfer_impl_, original_weight, packed_weight.GetUnowned()));

  const bool sharing_allowed = prepacked_weight_cache != nullptr;

  if (sharing_allowed) {
    std::array<void*, 1> buffer_data_ptrs = {weight_info.owned_data.get()};
    std::array<size_t, 1> buffer_data_sizes = {weight_info.num_bytes};

    Ort::UnownedSharedPrePackedWeightCache weight_cache(prepacked_weight_cache);

    // weight_cache takes ownership of the data. As the API documentation states, this requires that the
    // weight data is allocated with the OrtAllocator provided as a parameter to OrtKernelImpl::PrePackWeight.
    RETURN_IF_ERROR(weight_cache.StoreWeightData(buffer_data_ptrs.data(),
                                                 buffer_data_sizes.data(),
                                                 buffer_data_ptrs.size()));

    // IMPORTANT: This kernel no longer owns the packed weight data.
    // weight_info.shared_data will be initialized in the call to SetSharedPrePackedWeightImpl.
    weight_info.owned_data.release();
  }

  mul_kernel->packed_weight_1_info_ = std::move(weight_info);
  *is_packed = true;

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

/*static*/
OrtStatus* ORT_API_CALL Mul::SetSharedPrePackedWeightImpl(OrtKernelImpl* this_ptr,
                                                          const void* const* buffer_data_ptrs,
                                                          const size_t* buffer_data_sizes,
                                                          size_t num_buffers, int input_index) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  Mul* mul_kernel = static_cast<Mul*>(this_ptr);

  if (input_index != 1) {
    std::ostringstream oss;
    oss << "ExampleKernelEp did not expect a call to OrtKernelImpl::SetSharedPrePackedWeight for input index "
        << input_index << " of the Mul kernel.";
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, oss.str().c_str());
  }

  RETURN_IF(num_buffers != 1, Ort::GetApi(), "Invalid number of pre-packed data buffers for Mul kernel's 2nd input");
  RETURN_IF(!mul_kernel->packed_weight_1_info_.has_value(), Ort::GetApi(),
            "ERROR! OrtKernelImpl::PrePackWeight should have "
            "initialized a valid PackedWeightInfo struct for use in SetSharedPrePackedWeight.");

  // Check that the buffer size is what we expect.
  RETURN_IF(buffer_data_sizes[0] != mul_kernel->packed_weight_1_info_->num_bytes, Ort::GetApi(),
            "ExampleKernelEp received an unexpected buffer size in a call to OrtKernelImpl::SetSharedPrePackedWeight "
            "for the Mul kernel.");

  // Update buffer data pointer because the shared memory could potentially originate from a different
  // kernel instance.
  mul_kernel->packed_weight_1_info_->shared_data = buffer_data_ptrs[0];

  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

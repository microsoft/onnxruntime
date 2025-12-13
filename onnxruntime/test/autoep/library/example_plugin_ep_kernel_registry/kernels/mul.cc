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
  gsl::span<const float> input1;
  std::vector<int64_t> shape1;

  if (packed_weight_1_info_.has_value()) {
    shape1 = packed_weight_1_info_->shape;

    size_t num_elems = 1;
    for (auto s : shape1) {
      num_elems *= s;
    }

    input1 = gsl::span<const float>(reinterpret_cast<const float*>(packed_weight_1_info_->data.get()), num_elems);
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
}

OrtStatus* Mul::DoPrePackWeight(const OrtValue* tensor, int input_index, OrtAllocator* alloc,
                                OrtSharedPrePackedWeightCache* prepacked_weight_cache, /*out*/ bool& is_packed) {
  // This example Mul kernel does not really need to pre-pack mul initializers, but we show it here as an example.
  // This implementation just copies original tensor without modification. An actual implementation would, for example,
  // transform to an appropriate data layout.

  if (input_index != 1) {
    is_packed = false;
    return nullptr;
  }

  Ort::ConstValue original_weight(tensor);
  auto type_shape_info = original_weight.GetTensorTypeAndShapeInfo();
  size_t num_bytes = original_weight.GetTensorSizeInBytes();

  PackedWeightInfo weight_info = {};
  weight_info.mem_info = Ort::ConstMemoryInfo(alloc->Info(alloc));
  weight_info.shape = type_shape_info.GetShape();
  weight_info.elem_type = type_shape_info.GetElementType();
  weight_info.num_bytes = num_bytes;
  weight_info.data = AllocateBytes(alloc, num_bytes);

  // Note: This Ort::Value does not own the underlying data. It is owned by `alloc`.
  Ort::Value packed_weight = Ort::Value::CreateTensor(weight_info.mem_info, weight_info.data.get(),
                                                      weight_info.num_bytes, weight_info.shape.data(),
                                                      weight_info.shape.size(), weight_info.elem_type);

  RETURN_IF_ERROR(CopyTensor(original_weight, packed_weight.GetUnowned()));

  if (prepacked_weight_cache != nullptr) {
    std::array<void*, 1> buffer_data_ptrs = {weight_info.data.get()};
    std::array<size_t, 1> buffer_data_sizes = {weight_info.num_bytes};

    RETURN_IF_ERROR(Ort::GetEpApi().SharedPrePackedWeightCache_StoreWeightData(prepacked_weight_cache,
                                                                               buffer_data_ptrs.data(),
                                                                               buffer_data_sizes.data(),
                                                                               buffer_data_ptrs.size(),
                                                                               alloc));

    // IMPORTANT: This kernel no longer owns the packed weight data.
    // It will be re-initialized in the call to UseSharedPrePackWeight()
    weight_info.data.release();
    weight_info.data = nullptr;
  }

  packed_weight_1_info_ = std::move(weight_info);
  is_packed = true;
  return nullptr;
}

OrtStatus* Mul::DoSetSharedPrePackedWeight(const void* const* buffer_data_ptrs, size_t num_buffers,
                                           int input_index) {
  if (input_index != 1) {
    std::ostringstream oss;
    oss << "ExampleKernelEp did not expect a call to OrtKernelImpl::SetSharedPrePackedWeight for input index "
        << input_index << " of the Mul kernel.";
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, oss.str().c_str());
  }

  RETURN_IF(num_buffers != 1, Ort::GetApi(), "Invalid number of pre-packed data buffers for Mul kernel's 2nd input");
  RETURN_IF(!packed_weight_1_info_.has_value(), Ort::GetApi(),
            "ERROR! OrtKernelImpl::PrePackWeight should have "
            "initialized a valid PackedWeightInfo struct for use in SetSharedPrePackedWeight.");

  packed_weight_1_info_->data = AllocationUniquePtr(const_cast<void*>(buffer_data_ptrs[0]),
                                                    [](void* /*ptr*/) { /*no delete (don't own)*/ });

  return nullptr;
}

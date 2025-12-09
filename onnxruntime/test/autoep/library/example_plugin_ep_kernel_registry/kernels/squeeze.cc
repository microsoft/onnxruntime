// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "squeeze.h"

#include <gsl/span>
#include <cassert>

#include "utils.h"

// Support ONNX Squeeze versions 21, 23, and 24.
// Kernel creation functions are typically defined separately for new operator versions to account for things like new
// data types. One could technically support all three versions with a single call to
// ONNX_OPERATOR_VERSIONED_KERNEL_EX(Squeeze, kOnnxDomain, 21, 24, ...), but this example shows the more common usage.

// ONNX Squeeze version 21
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    /*start_version*/ 21, /*end_version (inclusive)*/ 22,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("axes", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddInputOutputAlias(0, 0)),
    Squeeze)

// ONNX Squeeze version 23
ONNX_OPERATOR_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    /*version*/ 23,  // Equivalent to start_version: 23, end_version: 23
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("axes", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddInputOutputAlias(0, 0)),
    Squeeze)

// ONNX Squeeze version 24.
ONNX_OPERATOR_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    /*version*/ 24,  // Equivalent start_version: 24, end_version: 24
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("axes", GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))
         .AddInputOutputAlias(0, 0)),
    Squeeze)

Squeeze::Squeeze(const OrtKernelInfo* info, void* state, PrivateTag) : BaseKernelImpl(info, state) {}

/*static*/
OrtStatus* Squeeze::Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Squeeze>& kernel) {
  Ort::ConstKernelInfo kernel_info(info);
  kernel = std::make_unique<Squeeze>(info, state, PrivateTag{});
  return nullptr;
}

static int64_t HandleNegativeAxis(int64_t axis, int64_t tensor_rank) {
  return axis < 0 ? axis + tensor_rank : axis;
}

static std::vector<int64_t> ComputeOutputShape(gsl::span<const int64_t> input_shape, gsl::span<const int64_t> axes) {
  size_t j = 0;
  std::vector<int64_t> output_shape;
  auto num_dimensions = input_shape.size();

  // Handle negative axis, then resort and uniq.
  std::vector<int64_t> axes_corrected(axes.size());
  for (size_t i = 0; i < axes.size(); i++) {
    axes_corrected[i] = HandleNegativeAxis(axes[i], num_dimensions);
  }
  std::sort(axes_corrected.begin(), axes_corrected.end());
  axes_corrected.erase(std::unique(axes_corrected.begin(), axes_corrected.end()), axes_corrected.end());

  for (size_t i = 0; i < num_dimensions; ++i) {
    if ((j < axes_corrected.size() && axes_corrected[j] == static_cast<int64_t>(i)) ||
        (axes_corrected.size() == 0 && input_shape[i] == 1)) {
      assert(input_shape[i] == 1);
      ++j;
      continue;
    }
    output_shape.push_back(input_shape[i]);
  }
  return output_shape;
}

OrtStatus* Squeeze::DoCompute(OrtKernelContext* kernel_ctx) {
  Ort::KernelContext kernel_context(kernel_ctx);
  static_cast<void>(this->state_);  // NOTE: Unused in this example.

  gsl::span<const float> input0;
  std::vector<int64_t> shape0;
  RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 0, input0, shape0));

  size_t num_inputs = kernel_context.GetInputCount();
  std::vector<int64_t> axes;

  if (num_inputs == 2) {
    // Axes is an explicit input.
    gsl::span<const int64_t> axes_input;
    std::vector<int64_t> axes_shape;
    RETURN_IF_ERROR(GetKernelInputDataAndShape<int64_t>(kernel_context, 1, axes_input, axes_shape));
    assert(axes_shape.size() == 1);

    axes.assign(axes_input.begin(), axes_input.end());
  }

  std::vector<int64_t> output_shape = ComputeOutputShape(shape0, axes);
  Ort::UnownedValue output = kernel_context.GetOutput(0, output_shape);
  float* output_data = output.GetTensorMutableData<float>();
  size_t num_bytes = output.GetTensorSizeInBytes();

  if (input0.data() != output_data) {  // Don't copy if src == dst
    // This uses a memcpy because the input and output are both located in the EP's device memory (i.e., cpu memory).
    // Normally, an EP would use a OrtDataTransferImpl to generically handle copies where the source and destination
    // could be on different devices.
    memcpy(output_data, input0.data(), num_bytes);
  }

  return nullptr;
}

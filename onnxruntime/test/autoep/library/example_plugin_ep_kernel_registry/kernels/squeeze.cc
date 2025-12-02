// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "squeeze.h"

#include <gsl/span>
#include <cassert>

#include "utils.h"

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    21, 24,
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
  const OrtEpApi& ep_api = Ort::GetEpApi();
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

  if (input0.data() != output_data) {
    std::array<const OrtValue*, 1> src_tensors = {kernel_context.GetInput(0)};
    std::array<OrtValue*, 1> dst_tensors = {output};

    RETURN_IF_ERROR(ep_api.KernelInfo_CopyTensors(info_,
                                                  src_tensors.data(),
                                                  dst_tensors.data(),
                                                  /*stream*/ nullptr,
                                                  src_tensors.size()));
  }

  return nullptr;
}

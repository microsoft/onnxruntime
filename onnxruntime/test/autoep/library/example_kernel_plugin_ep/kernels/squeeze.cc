// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "squeeze.h"

#include <gsl/span>
#include <cassert>

#include "utils.h"

ONNX_OPERATOR_KERNEL_EX(
    Squeeze,
    kOnnxDomain,
    13,
    (Ort::KernelDefBuilder()
         .AddTypeConstraint("T", MLDataTypes::GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT))
         .AddTypeConstraint("axes", MLDataTypes::GetTensorType(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64))),
    Squeeze)

Squeeze::Squeeze(const OrtKernelInfo* info, void* state, PrivateTag)
    : info_(info),
      state_(state) {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* Squeeze::Create(const OrtKernelInfo* info, void* state, /*out*/ std::unique_ptr<Squeeze>& kernel) {
  Ort::ConstKernelInfo kernel_info(info);

  try {
    size_t num_inputs = kernel_info.GetInputCount();
    if (num_inputs != 2) {
      Ort::Status status("Squeeze for example KernelEp only supports axes as an input (opset >= 13)", ORT_EP_FAIL);
      return status.release();
    }

    kernel = std::make_unique<Squeeze>(info, state, PrivateTag{});
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
OrtStatus* ORT_API_CALL Squeeze::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  Squeeze* squeeze = static_cast<Squeeze*>(this_ptr);
  return squeeze->DoCompute(kernel_ctx);
}

/*static*/
void ORT_API_CALL Squeeze::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<Squeeze*>(this_ptr);
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

OrtStatus* Squeeze::DoCompute(OrtKernelContext* kernel_ctx) noexcept {
  const OrtEpApi& ep_api = Ort::GetEpApi();
  Ort::KernelContext kernel_context(kernel_ctx);
  (void)this->state_;  // NOTE: Unused in this example.

  try {
    gsl::span<const float> input0;
    std::vector<int64_t> shape0;
    RETURN_IF_ERROR(GetKernelInputDataAndShape<float>(kernel_context, 0, input0, shape0));

    gsl::span<const int64_t> axes_input;
    std::vector<int64_t> axes_shape;
    RETURN_IF_ERROR(GetKernelInputDataAndShape<int64_t>(kernel_context, 1, axes_input, axes_shape));
    assert(axes_shape.size() == 1);

    std::vector<int64_t> output_shape = ComputeOutputShape(shape0, axes_input);
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
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }

  return nullptr;
}

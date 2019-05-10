// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif
#include "pad.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(Pad,
                        kMSDomain,
                        1,
                        kCpuExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        Pad<float>);

template <>
Status Pad<float>::Compute(OpKernelContext* ctx) const {
  const Tensor* input_tensor = ctx->Input<Tensor>(0);
  std::vector<int64_t> output_dims(input_tensor->Shape().GetDims());
  size_t dimension_count = input_tensor->Shape().NumDimensions();

  const auto& pads_tensor = *ctx->Input<Tensor>(1);
  const auto& pads_tensor_dims = pads_tensor.Shape().GetDims();
  ORT_ENFORCE(pads_tensor.DataType() == DataTypeImpl::GetType<int64_t>(),
              "Pads tensor should be an INT64 tensor");
  ORT_ENFORCE(pads_tensor_dims.size() == 1 || (pads_tensor_dims.size() == 2 && pads_tensor_dims[0] == 1),
              "Pads tensor should be a 1D tensor of shape [2 * input_rank] or a 2D tensor of shape [1, 2 * input_rank]");

  std::vector<int64_t> pads(2 * dimension_count, 0);
  const int64_t* pads_tensor_raw_data = pads_tensor.template Data<int64_t>();
  size_t pads_size = static_cast<size_t>(pads_tensor.Shape().Size());
  ORT_ENFORCE(pads_size == 2 * dimension_count,
              "Pads tensor size should be equal to twice the input dimension count ");

  for (size_t i = 0; i < pads_size; ++i) {
    pads[i] = pads_tensor_raw_data[i];
  }

  // Separate out any negative pads into the slices array
  std::vector<int64_t> slices(pads.size(), 0);
  for (size_t index = 0; index < pads.size(); index++) {
    if (pads[index] < 0) {
      slices[index] = pads[index];
      pads[index] = 0;
    }
  }

  float value = 0;
  const Tensor* value_tensor = ctx->Input<Tensor>(2);
  if (nullptr != value_tensor) {
    ORT_ENFORCE(value_tensor->DataType() == DataTypeImpl::GetType<float>() &&
                    value_tensor->Shape().Size() == 1,
                "Value tensor should be a 1D tensor of size 1 with the same type as that of the input tensor");
    value = value_tensor->template Data<float>()[0];
  }

  return PadCpuImpl<float>(ctx, pads, slices, mode_, value);
}

}  // namespace contrib
};  // namespace onnxruntime

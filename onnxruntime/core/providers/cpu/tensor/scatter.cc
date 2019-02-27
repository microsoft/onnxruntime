// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//https://github.com/onnx/onnx/blob/master/docs/Operators.md#Scatter
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {

class Scatter final : public OpKernel {
 public:
  Scatter(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(),
                "Missing/Invalid 'axis' attribute value");
  }
  ~Scatter() = default;
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

ONNX_CPU_OPERATOR_KERNEL(
    Scatter,
    9,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    Scatter);

template <class Tin>
Status CopyScatterData(const Tensor* data_input, const Tensor* indecies_input, const Tensor* updates_input,
                       const int64_t axis, Tensor* data_output) {
  const TensorShape& input_data_shape = data_input->Shape();
  const Tin* indices_data = indecies_input->template Data<Tin>();
  const auto num_indices = indecies_input->Shape().Size();
  for (int64_t i = 0; i < num_indices; ++i) {
    Tin idx = indices_data[i];
    if (idx < 0 || idx >= input_data_shape[axis]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "indices element out of data bounds, idx=", idx,
                             " data_dim=", input_data_shape[axis]);
    }
  }

  const auto input_elements = input_data_shape.Size();
  const auto element_bytes = data_input->DataType()->Size();
  const auto total_input_bytes = input_elements * element_bytes;

  const uint8_t* src_base = reinterpret_cast<const uint8_t*>(data_input->DataRaw());
  uint8_t* dst_base = reinterpret_cast<uint8_t*>(data_output->MutableDataRaw());
  const bool is_string_type = data_input->DataType() == DataTypeImpl::GetType<std::string>();

  // We allow runtime to re-use input for output. If input/output Tensor* are the same
  // we do not copy
  if (data_input != data_output) {
    if (is_string_type) {
      const std::string* str_begin = data_input->template Data<std::string>();
      const std::string* str_end = str_begin + input_elements;
      std::string* dst = data_output->template MutableData<std::string>();
      std::copy(str_begin, str_end, dst);
    } else {
      memcpy(dst_base, src_base, total_input_bytes);
    }
  }

  // Now poke updates
  const auto data_batches = input_data_shape.SizeToDimension(axis);
  const auto data_batch_bytes = input_data_shape.SizeFromDimension(axis) * element_bytes;
  const auto block = input_data_shape.SizeFromDimension(axis + 1);
  const auto block_bytes = block * element_bytes;

  // Initialize with zeros.
  const uint8_t* update_data = reinterpret_cast<const uint8_t*>(updates_input->DataRaw());
  for (int64_t batch = 0; batch < data_batches; ++batch) {
    const auto batch_bytes_offset = batch * data_batch_bytes;
    for (int64_t update_idx = 0; update_idx < num_indices; ++update_idx) {
      const Tin axis_idx = indices_data[update_idx];
      const auto block_offset = (block > 1) ? update_idx % block : 0;
      const auto dst_offset_bytes = batch_bytes_offset + axis_idx * block_bytes +
                                    block_offset * element_bytes;

      assert(dst_offset_bytes < total_input_bytes);
      if (is_string_type) {
        reinterpret_cast<std::string*>(dst_base)[dst_offset_bytes / element_bytes] =
            reinterpret_cast<const std::string*>(update_data)[update_idx];
      } else {
        // Copy the element
        auto src_offset_bytes = update_idx * element_bytes;
        memcpy(dst_base + dst_offset_bytes, update_data + src_offset_bytes, element_bytes);
      }
    }
  }
  return Status::OK();
}

Status Scatter::Compute(OpKernelContext* context) const {
  auto data_input = context->Input<Tensor>(0);
  const auto& input_data_shape = data_input->Shape();
  const auto axis = HandleNegativeAxis(axis_, input_data_shape.NumDimensions());

  auto indices_input = context->Input<Tensor>(1);
  auto updates_input = context->Input<Tensor>(2);

  if (data_input->DataType() != updates_input->DataType()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "data type is different from updates type");
  }

  auto& indicies_dims = indices_input->Shape().GetDims();
  auto& updates_dims = updates_input->Shape().GetDims();
  if (indicies_dims.size() != updates_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Indicies and updates must have the same number of dimensions");
  }

  for (size_t i = 0; i < indicies_dims.size(); ++i) {
    if (indicies_dims[i] != updates_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indicies vs updates dimensions differs at position=", i,
                             " ", indicies_dims[i], " vs ", updates_dims[i]);
    }
  }

  auto data_output = context->Output(0, input_data_shape);

  MLDataType Tind_type = indices_input->DataType();
  if (Tind_type == DataTypeImpl::GetType<int32_t>()) {
    return CopyScatterData<int32_t>(data_input, indices_input, updates_input, axis, data_output);
  } else if (Tind_type == DataTypeImpl::GetType<int64_t>()) {
    return CopyScatterData<int64_t>(data_input, indices_input, updates_input, axis, data_output);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expecting indices to be either int32_t or int64_t");
}

}  // namespace onnxruntime

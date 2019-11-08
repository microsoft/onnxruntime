// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//https://github.com/onnx/onnx/blob/master/docs/Operators.md#Scatter
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {

class Scatter final : public OpKernel {
 public:
  explicit Scatter(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(),
                "Missing/Invalid 'axis' attribute value");
  }

  ~Scatter() = default;
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Scatter,
    9, 10,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    Scatter);

ONNX_CPU_OPERATOR_KERNEL(
    ScatterElements,
    11,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::AllTensorTypes())
        .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(), DataTypeImpl::GetTensorType<int64_t>()}),
    Scatter);

template <class Tin, class Tdata>
Status CopyScatterData(const Tensor* data_input, const Tensor* indices_input, const Tensor* updates_input,
                       const int64_t axis, Tensor* data_output) {
  const TensorShape& input_data_shape = data_input->Shape();
  const Tin* indices_data_raw = indices_input->template Data<Tin>();
  const auto num_indices = indices_input->Shape().Size();

  std::vector<Tin> indices_data;
  indices_data.reserve(num_indices);

  auto axis_dim_limit = input_data_shape[axis];

  for (int64_t i = 0; i < num_indices; ++i) {
    Tin idx = indices_data_raw[i];

    if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "indices element out of data bounds, idx=", idx,
                             " must be within the inclusive range [", -axis_dim_limit,
                             ",", axis_dim_limit - 1, "]");
    }

    indices_data.push_back(idx < 0 ? idx + static_cast<Tin>(axis_dim_limit) : idx);
  }

  const auto input_elements = input_data_shape.Size();
  const auto total_input_bytes = data_input->SizeInBytes();

  const auto* src_base = static_cast<const Tdata*>(data_input->DataRaw());
  auto* dst_base = static_cast<Tdata*>(data_output->MutableDataRaw());
  bool is_string_type = utils::IsDataTypeString(data_input->DataType());

  // We allow runtime to re-use input for output. If input/output Tensor* are the same
  // we do not copy
  if (src_base != dst_base) {
    if (is_string_type) {
      const auto* str_begin = data_input->template Data<std::string>();
      const std::string* str_end = str_begin + input_elements;
      auto* dst = data_output->template MutableData<std::string>();
      std::copy(str_begin, str_end, dst);
    } else {
      memcpy(static_cast<void*>(dst_base), static_cast<const void*>(src_base), total_input_bytes);
    }
  }

  // Now poke updates

  const auto& upd_shape = updates_input->Shape();
  const auto num_dims = input_data_shape.NumDimensions();
  assert(num_dims > 0);

  // Allocate and zero out counts. The input/output is of the same rank as
  // indices/updates but the actual dimensions of indices/updates must be less or equal
  // than that of input/output because we can update no more elements than
  // the input contains. As we walk through the indices/updates
  // we maintain dimension count as we will need to use it
  // to compute output offset but using input/output dim values.
  // We treat the whole array as a number where each element having
  // different cardinality according to the upd_shape dimensions.
  // As each counter reaches its max (upd_shape) it resets to zero
  // and we carry to the more significant dim (right to left)
  std::vector<int64_t> dim_counters(num_dims);

  // This vector contains number of elements under the dimension.
  // For example, for the dimensions of [4, 2, 3] the vector
  // would contain [6, 3, 1] since for each count of dim 1 it
  // contains 3 elements of dim 2.
  // For each count of dim 0 we would have 2x3=6 elements.
  // The last value is always 1.
  // We use it to compute output element offset. For a given value of
  // counters we multiple each counter value per corresponding entry of dim_block_size value
  // and add up resulting the output element offset. However, for dimensions
  // that are equal to the specified axis value we take indices_data[index]
  // instead of the counter value.
  // E.g. for 3-dim and axis=0
  //    output[indices[i][j][k]][j][k] = updates[i][j][k]
  // for axis 1
  //    output[i][indices[i][j][k]][k] = updates[i][j][k]
  // and so on
  std::vector<int64_t> dim_block_size(num_dims);

  dim_block_size.back() = 1;
  if (num_dims > 1) {
    // We start at num_dims - 2 because we already pre-populated
    // the last element above
    for (auto i = int64_t(num_dims - 2); i >= 0; --i) {
      dim_block_size[i] = input_data_shape[i + 1] * dim_block_size[i + 1];
    }
  }

  const auto* update_data = static_cast<const Tdata*>(updates_input->DataRaw());
  // For every update we compute the destination offset and copy it there
  for (int64_t index = 0; index < num_indices;) {
    const Tin axis_idx = indices_data[index];

    // Compute the offset
    // See comments above for dim_block_size
    size_t dst_offset = 0;
    for (size_t i = 0; i < num_dims; ++i) {
      if (i == size_t(axis)) {
        // replace the counter with the update index for this dim
        dst_offset += axis_idx * dim_block_size[i];
      } else {
        dst_offset += dim_counters[i] * dim_block_size[i];
      }
    }

    dst_base[dst_offset] = update_data[index];

    if (++index == num_indices) {
      break;
    }
    // Increment counters
    // See comments for dim_counters above
    for (auto i = int64_t(num_dims - 1); i >= 0; --i) {
      auto v = ++dim_counters[i];
      assert(v <= upd_shape[i]);
      if (v < upd_shape[i]) {
        // No carry, done
        break;
      }
      // No carry for the most significant dim
      assert(i > 0);
      dim_counters[i] = 0;
    }
  }
  return Status::OK();
}

template <class T, class... Args>
inline Status CopyInt32Index(Args&&... args) {
  return CopyScatterData<int32_t, T>(std::forward<Args>(args)...);
}

template <class T, class... Args>
inline Status CopyInt64Index(Args&&... args) {
  return CopyScatterData<int64_t, T>(std::forward<Args>(args)...);
}

Status Scatter::Compute(OpKernelContext* context) const {
  const auto* data_input = context->Input<Tensor>(0);
  const auto& input_data_shape = data_input->Shape();
  const auto axis = HandleNegativeAxis(axis_, input_data_shape.NumDimensions());

  const auto* indices_input = context->Input<Tensor>(1);
  const auto* updates_input = context->Input<Tensor>(2);

  if (data_input->DataType() != updates_input->DataType()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "data type is different from updates type");
  }

  auto& indices_dims = indices_input->Shape().GetDims();
  auto& updates_dims = updates_input->Shape().GetDims();
  if (indices_dims.size() != updates_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Indices and updates must have the same rank");
  }

  for (size_t i = 0; i < indices_dims.size(); ++i) {
    if (indices_dims[i] != updates_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices vs updates dimensions differs at position=", i,
                             " ", indices_dims[i], " vs ", updates_dims[i]);
    }
  }

  // According to the spec the rank of ind/upd shall be the same as input(data)
  // and we also want to make sure that the dimensions of the of the ind/upd do not
  // exceed that of the input
  auto& input_dims = input_data_shape.GetDims();
  if (input_dims.size() != indices_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices must have the same rank as Input. Indices rank=",
                           indices_dims.size(), ". Input rank=", input_dims.size());
  }

  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (input_dims[i] < indices_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices dim=", indices_dims[i], " at pos=", i,
                             " is greater than input dim=", input_dims[i]);
    }
  }

  auto* data_output = context->Output(0, input_data_shape);

  MLDataType Tind_type = indices_input->DataType();
  MLDataType Tdata_type = data_input->DataType();
  Status status;
  if (utils::IsPrimitiveDataType<int32_t>(Tind_type)) {
    DispatchOnTensorTypeWithReturn(Tdata_type, status, CopyInt32Index, data_input, indices_input, updates_input, axis, data_output);
  } else if (utils::IsPrimitiveDataType<int64_t>(Tind_type)) {
    DispatchOnTensorTypeWithReturn(Tdata_type, status, CopyInt64Index, data_input, indices_input, updates_input, axis, data_output);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Expecting indices to be either int32_t or int64_t");
  }
  return status;
}

}  // namespace onnxruntime

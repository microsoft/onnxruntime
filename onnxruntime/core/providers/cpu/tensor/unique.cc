// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/unique.h"
#include <map>
#include <core/common/safeint.h>
#include "core/common/gsl.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/providers/common.h"
#include "core/providers/op_kernel_type_control.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPES_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Unique, Input, 0,
    float, int64_t, int8_t, std::string, double);
}

using EnabledUniqueDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Unique, Input, 0);

/*
ONNX_OPERATOR_SET_SCHEMA(
    Unique,
    11,
    OpSchema()
        .SetDoc(Unique_ver11_doc)
        .Attr(
            "sorted",
            "(Optional) Whether to sort the unique elements in ascending order before returning as output. "
            "Must be one of 0, or 1 (default).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "axis",
            "(Optional) The dimension to apply unique. If not specified, the unique elements of the "
            "flattened input are returned. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            OPTIONAL)
        .Input(0, "X", "A N-D input tensor that is to be processed.", "T")
        .Output(
            0,
            "Y",
            "A tensor of the same type as 'X' "
            "containing all the unique values or subtensors sliced along a provided 'axis' in 'X', either sorted "
            "or maintained in the same order they occur in input 'X'",
            "T")
        .Output(
            1,
            "indices",
            "A 1-D INT64 tensor "
            "containing indices of 'Y' elements' first occurance in 'X'. "
            "When 'axis' is provided, it contains indices to subtensors in input 'X' on the 'axis'. "
            "When 'axis' is not provided, it contains indices to values in the flattened input tensor. ",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(
            2,
            "inverse_indices",
            "A 1-D INT64 tensor "
            "containing, for elements of 'X', its corresponding indices in 'Y'. "
            "When 'axis' is provided, it contains indices to subtensors in output 'Y' on the 'axis'. "
            "When 'axis' is not provided, it contains indices to values in output 'Y'. ",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(
            3,
            "counts",
            "A 1-D INT64 tensor containing "
            "the count of each element "
            "of 'Y' in input 'X'",
            "tensor(int64)",
            OpSchema::Optional)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Input can be of any tensor type.")
*/
ONNX_CPU_OPERATOR_KERNEL(
    Unique,
    11,
    KernelDefBuilder().TypeConstraint("T",
                                      BuildKernelDefConstraintsFromTypeList<EnabledUniqueDataTypes>()),
    Unique);

Status Unique::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);

  Status status;
  // arbitrary set of types to support initially
  // Note: The non-string implementations can probably be based on data type size.
  if (input.IsDataType<double>())
    status = ComputeImpl<double>(*context);
  else if (input.IsDataType<float>())
    status = ComputeImpl<float>(*context);
  else if (input.IsDataType<int64_t>())
    status = ComputeImpl<int64_t>(*context);
  else if (input.IsDataType<int8_t>())
    status = ComputeImpl<int8_t>(*context);
  else if (input.IsDataTypeString())
    status = ComputeImpl<std::string>(*context);
  else
    status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported tensor type of ", input.DataType());

  return status;
}

// class to represent a subtensor along a given axis for a single entry on that axis
template <typename T>
class Subtensor {
 public:
  // Create Subtensor for entry 'idx' on axis 'axis'
  // n_axis is the number of entries for 'axis' is the original data.
  //   e.g. if original shape was [4, 2] and axis is 1, n_axis == 2.
  // subtensor_shape is the shape for the subtensor. the dimension value for the 'axis' dimension in it will be 1.
  Subtensor(const gsl::span<const T>& data, const TensorShape& subtensor_shape,
            int64_t axis, int64_t n_axis, int64_t idx) {
    // rows and columns for the slice along axis, flattened to 2D by merging the dimensions before and after the axis
    int64_t columns = subtensor_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis));
    int64_t rows = subtensor_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis));
    items_.reserve(SafeInt<size_t>(rows) * columns);
    size_t cur_data = SafeInt<size_t>(idx) * columns;  // offset into data for first row of slice

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < columns; ++c) {
        // TODO: could copy blocks instead of individual items for non std::string types
        items_.push_back(data[cur_data + c]);
      }

      cur_data += SafeInt<size_t>(columns) * n_axis;
    }
  }

  bool operator<(const Subtensor& rhs) const {
    return items_ < rhs.items_;
  }

  const std::vector<T>& GetItems() const { return items_; }

 private:
  // TODO: Simple copy for now. std::string would be better as std::reference_wrapper<std::string>
  std::vector<T> items_;
};

template <typename T>
static void CreateFlattenedOutput(OpKernelContext& context,
                                  const std::map<const T, int64_t>& offsets,         // map sorted key to unsorted idx
                                  const std::vector<std::vector<int64_t>>& indices,  // unsorted
                                  const std::vector<int64_t>& inverse_index,         // unsorted
                                  bool sorted) {
  int64_t num_unique = static_cast<int64_t>(indices.size());
  Tensor& Y = *context.Output(0, {num_unique});
  Tensor* indices_out = context.Output(1, {num_unique});
  Tensor* inverse_indices = context.Output(2, {static_cast<int64_t>(inverse_index.size())});
  Tensor* counts = context.Output(3, {num_unique});

  auto Y_data = Y.MutableDataAsSpan<T>();
  gsl::span<int64_t> indices_data = indices_out != nullptr ? indices_out->MutableDataAsSpan<int64_t>()
                                                           : gsl::span<int64_t>();
  gsl::span<int64_t> inverse_indices_data = inverse_indices != nullptr ? inverse_indices->MutableDataAsSpan<int64_t>()
                                                                       : gsl::span<int64_t>();
  gsl::span<int64_t> counts_data = counts != nullptr ? counts->MutableDataAsSpan<int64_t>()
                                                     : gsl::span<int64_t>();

  // iterate using 'offsets' which is sorted, but contains the offset of the unsorted entry
  auto offsets_iter = offsets.begin();
  for (int64_t i = 0, end = num_unique; i < end; ++i, ++offsets_iter) {
    // write sequentially if we want sorted output, use the unsorted_idx if not
    auto unsorted_idx = offsets_iter->second;
    auto output_idx = sorted ? i : unsorted_idx;

    Y_data[onnxruntime::narrow<size_t>(output_idx)] = offsets_iter->first;

    if (indices_out) {
      indices_data[onnxruntime::narrow<size_t>(output_idx)] = indices[onnxruntime::narrow<size_t>(unsorted_idx)].front();
    }

    if (counts) {
      counts_data[onnxruntime::narrow<size_t>(output_idx)] = indices[onnxruntime::narrow<size_t>(unsorted_idx)].size();
    }
  }

  if (inverse_indices) {
    if (sorted) {
      // need to convert unsorted entries in the inverse index to their sorted values
      std::vector<int64_t> unsorted_to_sorted;
      unsorted_to_sorted.resize(onnxruntime::narrow<size_t>(num_unique));
      int64_t sorted_idx = 0;
      for (const auto& offset : offsets) {
        unsorted_to_sorted[onnxruntime::narrow<size_t>(offset.second)] = sorted_idx++;
      }

      for (size_t i = 0, end = inverse_index.size(); i < end; ++i) {
        inverse_indices_data[onnxruntime::narrow<size_t>(i)] = unsorted_to_sorted[onnxruntime::narrow<size_t>(inverse_index[i])];
      }
    } else {
      for (size_t i = 0, end = inverse_index.size(); i < end; ++i) {
        inverse_indices_data[i] = inverse_index[i];
      }
    }
  }
}

template <typename T>
static void CreateOutput(OpKernelContext& context,
                         const TensorShape& subtensor_shape,
                         int64_t axis,
                         const std::map<const Subtensor<T>, int64_t>& offsets,  // map sorted key to unsorted idx
                         const std::vector<std::vector<int64_t>>& indices,      // unsorted
                         const std::vector<int64_t>& inverse_index,             // unsorted
                         bool sorted) {
  int64_t num_unique = static_cast<int64_t>(indices.size());

  // rows and columns for the slice along axis, flattened to 2D by merging the dimensions before and after the axis
  int64_t num_cols = subtensor_shape.SizeFromDimension(onnxruntime::narrow<size_t>(axis));
  int64_t num_rows = subtensor_shape.SizeToDimension(onnxruntime::narrow<size_t>(axis));

  auto subtensor_dims = subtensor_shape.GetDims();
  std::vector<int64_t> Y_dims;
  Y_dims.reserve(subtensor_dims.size());
  for (int64_t i = 0, end = subtensor_dims.size(); i < end; ++i) {
    if (i == axis)
      Y_dims.push_back(num_unique);
    else
      Y_dims.push_back(subtensor_dims[onnxruntime::narrow<size_t>(i)]);
  }

  Tensor& Y = *context.Output(0, TensorShape(std::move(Y_dims)));
  Tensor* indices_out = context.Output(1, {num_unique});
  Tensor* inverse_indices = context.Output(2, {static_cast<int64_t>(inverse_index.size())});
  Tensor* counts = context.Output(3, {num_unique});

  auto Y_data = Y.MutableDataAsSpan<T>();
  gsl::span<int64_t> indices_data = indices_out != nullptr ? indices_out->MutableDataAsSpan<int64_t>()
                                                           : gsl::span<int64_t>();
  gsl::span<int64_t> inverse_indices_data = inverse_indices != nullptr ? inverse_indices->MutableDataAsSpan<int64_t>()
                                                                       : gsl::span<int64_t>();
  gsl::span<int64_t> counts_data = counts != nullptr ? counts->MutableDataAsSpan<int64_t>()
                                                     : gsl::span<int64_t>();

  // iterate using 'offsets' which is sorted, but contains the offset of the unsorted entry
  auto offsets_iter = offsets.begin();

  for (int64_t i = 0, end = num_unique; i < end; ++i, ++offsets_iter) {
    // write sequentially if we want sorted output, use the unsorted_idx if not
    auto unsorted_idx = offsets_iter->second;
    auto output_idx = (sorted ? i : unsorted_idx);

    const auto& items = offsets_iter->first.GetItems();
    auto item = items.cbegin();
    assert(static_cast<int64_t>(items.size()) == num_rows * num_cols);

    int64_t out_offset = output_idx * num_cols;

    for (int64_t row = 0; row < num_rows; ++row) {
      // copy num_cols items from entries to output
      if (std::is_same<T, std::string>::value) {
        std::copy(item, item + onnxruntime::narrow<size_t>(num_cols), &Y_data[onnxruntime::narrow<size_t>(out_offset)]);
      } else {
        std::copy_n(item, onnxruntime::narrow<size_t>(num_cols), &Y_data[onnxruntime::narrow<size_t>(out_offset)]);
      }

      item += onnxruntime::narrow<size_t>(num_cols);
      out_offset += num_unique * num_cols;
    }

    assert(item == items.cend());

    if (indices_out) {
      indices_data[onnxruntime::narrow<size_t>(output_idx)] = indices[onnxruntime::narrow<size_t>(unsorted_idx)].front();
    }

    if (counts) {
      counts_data[onnxruntime::narrow<size_t>(output_idx)] = indices[onnxruntime::narrow<size_t>(unsorted_idx)].size();
    }
  }

  if (inverse_indices) {
    if (sorted) {
      // need to convert unsorted entries in the inverse index to their sorted values
      std::vector<int64_t> unsorted_to_sorted;
      unsorted_to_sorted.resize(onnxruntime::narrow<size_t>(num_unique));
      int64_t sorted_idx = 0;
      for (const auto& offset : offsets) {
        unsorted_to_sorted[onnxruntime::narrow<size_t>(offset.second)] = sorted_idx++;
      }

      for (size_t i = 0, end = inverse_index.size(); i < end; ++i) {
        inverse_indices_data[i] = unsorted_to_sorted[onnxruntime::narrow<size_t>(inverse_index[i])];
      }
    } else {
      for (size_t i = 0, end = inverse_index.size(); i < end; ++i) {
        inverse_indices_data[i] = inverse_index[i];
      }
    }
  }
}

template <typename T>
Status Unique::ComputeImpl(OpKernelContext& context) const {
  if (!utils::HasType<EnabledUniqueDataTypes, T>()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Data type is not supported in this build.");
  }

  const Tensor& input = *context.Input<Tensor>(0);
  auto data = input.DataAsSpan<T>();

  if (flatten_) {
    std::map<const T, int64_t> offsets;  // offset of entry in indices. provides map between sorted and unsorted values
    std::vector<std::vector<int64_t>> indices;
    std::vector<int64_t> inverse_index;

    indices.reserve(data.size() / 2);  // arbitrary value. at worst 1 realloc but could be too large
    inverse_index.reserve(data.size());

    int64_t num_unique = 0;

    for (int64_t i = 0, end = input.Shape().Size(); i < end; ++i) {
      auto entry = offsets.find(data[onnxruntime::narrow<size_t>(i)]);
      if (entry == offsets.end()) {
        offsets[data[onnxruntime::narrow<size_t>(i)]] = num_unique;
        inverse_index.push_back({num_unique});
        indices.push_back({i});
        ++num_unique;
      } else {
        size_t indices_idx = onnxruntime::narrow<size_t>(entry->second);
        indices[indices_idx].push_back(i);
        inverse_index.push_back(onnxruntime::narrow<int64_t>(indices_idx));
      }
    }

    CreateFlattenedOutput(context, offsets, indices, inverse_index, sort_);
  } else {
    const auto& input_shape = input.Shape();
    const int64_t input_dims = static_cast<int64_t>(input_shape.NumDimensions());
    const int64_t axis = HandleNegativeAxis(axis_, input_dims);

    std::vector<int64_t> subtensor_dims;
    subtensor_dims.reserve(onnxruntime::narrow<size_t>(input_dims));
    for (int64_t i = 0; i < input_dims; ++i) {
      subtensor_dims.push_back(i == axis ? 1 : input_shape[onnxruntime::narrow<size_t>(i)]);
    }

    TensorShape subtensor_shape(std::move(subtensor_dims));

    std::map<const Subtensor<T>, int64_t> offsets;
    std::vector<std::vector<int64_t>> indices;
    std::vector<int64_t> inverse_index;

    indices.reserve(data.size() / 2);  // arbitrary value. at worst 1 realloc but could be too large
    inverse_index.reserve(data.size());

    int64_t num_unique = 0;
    int64_t n_axis = input_shape[onnxruntime::narrow<size_t>(axis)];

    for (int64_t i = 0; i < n_axis; ++i) {
      Subtensor<T> s(data, subtensor_shape, axis, n_axis, i);

      auto entry = offsets.find(s);
      if (entry == offsets.end()) {
        offsets[std::move(s)] = num_unique;
        inverse_index.push_back({num_unique});
        indices.push_back({i});
        ++num_unique;
      } else {
        size_t indices_idx = onnxruntime::narrow<size_t>(entry->second);
        indices[indices_idx].push_back(i);
        inverse_index.push_back(onnxruntime::narrow<int64_t>(indices_idx));
      }
    }

    CreateOutput(context, subtensor_shape, axis, offsets, indices, inverse_index, sort_);
  }

  return Status::OK();
}

}  // namespace onnxruntime

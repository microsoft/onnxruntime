// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scatter
#include <type_traits>
#include <core/common/safeint.h>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/providers/common.h"
#include "core/providers/op_kernel_type_control.h"
#if defined(ENABLE_TRAINING_OPS)
#include "orttraining/training_ops/cpu/tensor/gather_elements_grad_impl.h"
#endif

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Scatter, Input, 0, element_type_lists::All);

ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterElements, Input, 0, element_type_lists::All);
}  // namespace op_kernel_type_control

using EnabledScatterDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, Scatter, Input, 0);

using EnabledScatterElementsDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterElements, Input, 0);

template <typename EnabledDataTypes>
class Scatter final : public OpKernel {
 public:
  explicit Scatter(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("axis", &axis_).IsOK(),
                "Missing/Invalid 'axis' attribute value");

    // 'reduction' attribute was added in opset 16.
    // its default value is 'none' in which case the op behaves the same as before opset 16.
    if (!info.GetAttr<std::string>("reduction", &reduction_).IsOK()) {
      reduction_ = "none";
    }
  }

  ~Scatter() = default;
  Status Compute(OpKernelContext* context) const override;

 private:
  int64_t axis_;
  std::string reduction_;
};

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Scatter,
    9, 10,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()),
    Scatter<EnabledScatterDataTypes>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterElements,
    11,
    12,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterElementsDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()),
    Scatter<EnabledScatterElementsDataTypes>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterElements,
    13,
    15,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterElementsDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()),
    Scatter<EnabledScatterElementsDataTypes>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterElements,
    16,
    17,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterElementsDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()),
    Scatter<EnabledScatterElementsDataTypes>);

ONNX_CPU_OPERATOR_KERNEL(
    ScatterElements,
    18,
    KernelDefBuilder()
        .MayInplace(0, 0)
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterElementsDataTypes>())
        .TypeConstraint("Tind", BuildKernelDefConstraints<int32_t, int64_t>()),
    Scatter<EnabledScatterElementsDataTypes>);

template <class T>
struct Func_Assignment {
  void operator()(T* a, const T* b) const {
    (*a) = (*b);
  }
};

template <class T>
struct Func_Add {
  void operator()(T* a, const T* b) const {
    (*a) += (*b);
  }
};

template <>
struct Func_Add<bool> {
  void operator()(bool* a, const bool* b) const {
    (*a) |= (*b);
  }
};

template <>
struct Func_Add<MLFloat16> {
  void operator()(MLFloat16*, const MLFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: MLFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'add'.");
  }
};

template <>
struct Func_Add<BFloat16> {
  void operator()(BFloat16*, const BFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'add'.");
  }
};

template <class T>
struct Func_Mul {
  void operator()(T* a, const T* b) const {
    (*a) *= (*b);
  }
};

template <>
struct Func_Mul<bool> {
  void operator()(bool* a, const bool* b) const {
    (*a) &= (*b);
  }
};

template <>
struct Func_Mul<std::string> {
  void operator()(std::string*, const std::string*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterElements opset 16 when reduction is 'mul'.");
  }
};

template <>
struct Func_Mul<MLFloat16> {
  void operator()(MLFloat16*, const MLFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: MLFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'mul'.");
  }
};

template <>
struct Func_Mul<BFloat16> {
  void operator()(BFloat16*, const BFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterElements opset 16 when reduction is 'mul'.");
  }
};

template <class T>
struct Func_Min {
  void operator()(T* a, const T* b) const {
    (*a) = (*a) < (*b) ? (*a) : (*b);
  }
};

template <>
struct Func_Min<bool> {
  void operator()(bool*, const bool*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: bool data type is not supported with ScatterElements opset 18 when reduction is 'min'.");
  }
};

template <>
struct Func_Min<std::string> {
  void operator()(std::string*, const std::string*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterElements opset 18 when reduction is 'min'.");
  }
};

template <>
struct Func_Min<BFloat16> {
  void operator()(BFloat16*, const BFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterElements opset 18 when reduction is 'min'.");
  }
};

template <class T>
struct Func_Max {
  void operator()(T* a, const T* b) const {
    (*a) = (*a) > (*b) ? (*a) : (*b);
  }
};

template <>
struct Func_Max<bool> {
  void operator()(bool*, const bool*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: bool data type is not supported with ScatterElements opset 18 when reduction is 'max'.");
  }
};

template <>
struct Func_Max<std::string> {
  void operator()(std::string*, const std::string*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterElements opset 18 when reduction is 'max'.");
  }
};

template <>
struct Func_Max<BFloat16> {
  void operator()(BFloat16*, const BFloat16*) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterElements opset 18 when reduction is 'max'.");
  }
};

template <class TIndex>
Status GetIndices(
    const Tensor& data_input, const Tensor& indices_input, int64_t axis,
    std::vector<int64_t>& indices_data) {
  const auto& input_data_shape = data_input.Shape();
  const auto* indices_data_raw = indices_input.Data<TIndex>();
  const auto num_indices = indices_input.Shape().Size();
  const auto axis_dim_limit = input_data_shape[narrow<size_t>(axis)];

  std::vector<int64_t> indices_data_result;
  indices_data_result.reserve(narrow<size_t>(num_indices));

  for (int64_t i = 0; i < num_indices; ++i) {
    const int64_t idx = static_cast<int64_t>(indices_data_raw[i]);

    if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "indices element out of data bounds, idx=", idx,
                             " must be within the inclusive range [", -axis_dim_limit,
                             ",", axis_dim_limit - 1, "]");
    }

    indices_data_result.push_back(idx < 0 ? idx + axis_dim_limit : idx);
  }

  indices_data = std::move(indices_data_result);
  return Status::OK();
}

template <class Tdata, typename FuncT>
Status ScatterData(
    const FuncT& func,
    const Tensor* data_input, const std::vector<int64_t>& indices_data, const Tensor* updates_input, int64_t axis,
    Tensor* data_output) {
  const TensorShape& input_data_shape = data_input->Shape();

  const auto input_elements = input_data_shape.Size();
  const auto total_input_bytes = data_input->SizeInBytes();

  const auto num_indices = narrow<int64_t>(indices_data.size());

  const auto* src_base = static_cast<const Tdata*>(data_input->DataRaw());
  auto* dst_base = static_cast<Tdata*>(data_output->MutableDataRaw());

  // We allow runtime to re-use input for output. If input/output Tensor* are the same
  // we do not copy
  if (src_base != dst_base) {
    if (std::is_same<Tdata, std::string>::value) {
      const auto* str_begin = data_input->Data<std::string>();
      const std::string* str_end = str_begin + input_elements;
      auto* dst = data_output->MutableData<std::string>();
      std::copy(str_begin, str_end, dst);
    } else {
      memcpy(static_cast<void*>(dst_base), static_cast<const void*>(src_base), total_input_bytes);
    }
  }

  // Now poke updates

  const auto& upd_shape = updates_input->Shape();
  const auto num_dims = input_data_shape.NumDimensions();
  ORT_RETURN_IF_NOT(num_dims > 0, "ScatterElements op: input tensor must have at least one dimension");

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
      dim_block_size[narrow<size_t>(i)] = input_data_shape[SafeInt<size_t>(i) + 1] * dim_block_size[SafeInt<size_t>(i) + 1];
    }
  }

  const auto* update_data = static_cast<const Tdata*>(updates_input->DataRaw());
  // For every update we compute the destination offset and copy it there
  for (int64_t index = 0; index < num_indices;) {
    const auto axis_idx = indices_data[narrow<size_t>(index)];

    // Compute the offset
    // See comments above for dim_block_size
    size_t dst_offset = 0;
    for (size_t i = 0; i < num_dims; ++i) {
      if (i == size_t(axis)) {
        // replace the counter with the update index for this dim
        dst_offset += narrow<size_t>(axis_idx * dim_block_size[narrow<size_t>(i)]);
      } else {
        dst_offset += narrow<size_t>(dim_counters[narrow<size_t>(i)] * dim_block_size[narrow<size_t>(i)]);
      }
    }

    func(dst_base + dst_offset, update_data + index);

    if (++index == num_indices) {
      break;
    }
    // Increment counters
    // See comments for dim_counters above
    for (auto i = int64_t(num_dims - 1); i >= 0; --i) {
      auto v = ++dim_counters[narrow<size_t>(i)];
      assert(v <= upd_shape[narrow<size_t>(i)]);
      if (v < upd_shape[narrow<size_t>(i)]) {
        // No carry, done
        break;
      }
      // No carry for the most significant dim
      assert(i > 0);
      dim_counters[narrow<size_t>(i)] = 0;
    }
  }
  return Status::OK();
}

template <typename TData>
struct ScatterDataDispatchTarget {
  Status operator()(const Tensor* data_input, const std::vector<int64_t>& indices_data, const Tensor* updates_input, int64_t axis,
                    const std::string& reduction, Tensor* data_output) const {
    if (reduction == "add")
      return ScatterData<TData>(
          Func_Add<TData>(), data_input, indices_data, updates_input, axis, data_output);
    else if (reduction == "mul")
      return ScatterData<TData>(
          Func_Mul<TData>(), data_input, indices_data, updates_input, axis, data_output);
    else if (reduction == "min")
      return ScatterData<TData>(
          Func_Min<TData>(), data_input, indices_data, updates_input, axis, data_output);
    else if (reduction == "max")
      return ScatterData<TData>(
          Func_Max<TData>(), data_input, indices_data, updates_input, axis, data_output);
    else  // if (reduction == "none")
      return ScatterData<TData>(
          Func_Assignment<TData>(), data_input, indices_data, updates_input, axis, data_output);
  }
};

template <typename EnabledDataTypes>
Status Scatter<EnabledDataTypes>::Compute(OpKernelContext* context) const {
  const auto* data_input = context->Input<Tensor>(0);
  const auto& input_data_shape = data_input->Shape();
  const auto axis = HandleNegativeAxis(axis_, input_data_shape.NumDimensions());

  const auto* indices_input = context->Input<Tensor>(1);
  const auto* updates_input = context->Input<Tensor>(2);

  if (data_input->DataType() != updates_input->DataType()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "data type is different from updates type");
  }

  auto indices_dims = indices_input->Shape().GetDims();
  auto updates_dims = updates_input->Shape().GetDims();
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
  auto input_dims = input_data_shape.GetDims();
  if (input_dims.size() != indices_dims.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices must have the same rank as Input. Indices rank=",
                           indices_dims.size(), ". Input rank=", input_dims.size());
  }

  for (size_t i = 0; i < input_dims.size(); ++i) {
    // For all axes except the axis of interest, make sure that the corresponding 'indices' shape
    // value is within bounds of the corresponding 'data' shape.
    if (static_cast<int64_t>(i) != axis && input_dims[i] < indices_dims[i]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Indices dim=", indices_dims[i], " at pos=", i,
                             " is greater than input dim=", input_dims[i]);
    }
  }

  Status status{};
  const auto index_type = indices_input->GetElementType();
  std::vector<int64_t> indices_data{};

  if (index_type == utils::ToTensorProtoElementType<int32_t>()) {
    status = GetIndices<int32_t>(*data_input, *indices_input, axis, indices_data);
  } else if (index_type == utils::ToTensorProtoElementType<int64_t>()) {
    status = GetIndices<int64_t>(*data_input, *indices_input, axis, indices_data);
  } else {
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Indices type is not supported.");
  }

  if (!status.IsOK()) {
    return status;
  }

  auto* data_output = context->Output(0, input_data_shape);
  const auto data_type = data_input->GetElementType();

  utils::MLTypeCallDispatcherFromTypeList<EnabledDataTypes> dispatcher{data_type};
  status = dispatcher.template InvokeRet<Status, ScatterDataDispatchTarget>(
      data_input, indices_data, updates_input, axis, this->reduction_, data_output);

  return status;
}

#if defined(ENABLE_TRAINING_OPS)

namespace contrib {

template <class T>
struct Func_Add {
  void operator()(T* a, const T* b) const {
    *a = *a + *b;
  }
};

template <class Tin, class Tdata>
Status GatherElementsGradImpl(const Tensor* indices_input, const Tensor* updates_input,
                              const int64_t axis, Tensor* data_output) {
  std::vector<int64_t> indices_data{};
  ORT_RETURN_IF_ERROR(GetIndices<Tin>(*data_output, *indices_input, axis, indices_data));
  return ScatterData<Tdata>(Func_Add<Tdata>(), data_output, indices_data, updates_input, axis, data_output);
}

#define GATHER_ELEMENTS_GRAD_IMPL_SPECIALIZED(Tin, Tdata) \
  template Status GatherElementsGradImpl<Tin, Tdata>(     \
      const Tensor* indices_input,                        \
      const Tensor* updates_input,                        \
      const int64_t axis,                                 \
      Tensor* data_output)

#define GATHER_ELEMENTS_GRAD_IMPL_TDATA_SPECIALIZED(Tdata) \
  GATHER_ELEMENTS_GRAD_IMPL_SPECIALIZED(int32_t, Tdata);   \
  GATHER_ELEMENTS_GRAD_IMPL_SPECIALIZED(int64_t, Tdata);

GATHER_ELEMENTS_GRAD_IMPL_TDATA_SPECIALIZED(float)
GATHER_ELEMENTS_GRAD_IMPL_TDATA_SPECIALIZED(double)

}  // namespace contrib

#endif

}  // namespace onnxruntime

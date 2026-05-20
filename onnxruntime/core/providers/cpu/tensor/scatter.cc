// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scatter
#include <atomic>
#include <type_traits>
#include <core/common/safeint.h>

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/platform/threadpool.h"
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
    concurrency::ThreadPool* tp,
    std::vector<int64_t>& indices_data) {
  const auto& input_data_shape = data_input.Shape();
  const auto* indices_data_raw = indices_input.Data<TIndex>();
  const auto num_indices = indices_input.Shape().Size();
  const auto axis_dim_limit = input_data_shape[narrow<size_t>(axis)];

  indices_data.resize(narrow<size_t>(num_indices));

  // When multiple indices are out-of-bounds, the reported index is nondeterministic
  // (whichever thread wins the CAS). This is acceptable—we only need to report that
  // validation failed and provide one example of a bad index.
  std::atomic<bool> found_error{false};
  std::atomic<int64_t> first_bad_idx{0};

  concurrency::ThreadPool::TryParallelFor(
      tp, narrow<std::ptrdiff_t>(num_indices), 1.0,
      [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        for (std::ptrdiff_t i = first; i < last; ++i) {
          const int64_t idx = static_cast<int64_t>(indices_data_raw[i]);
          if (idx < -axis_dim_limit || idx >= axis_dim_limit) {
            bool expected = false;
            if (found_error.compare_exchange_strong(expected, true)) {
              first_bad_idx.store(idx, std::memory_order_relaxed);
            }
            return;
          }
          indices_data[narrow<size_t>(i)] = idx < 0 ? idx + axis_dim_limit : idx;
        }
      });

  if (found_error.load()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "indices element out of data bounds, idx=", first_bad_idx.load(),
                           " must be within the inclusive range [", -axis_dim_limit,
                           ",", axis_dim_limit - 1, "]");
  }

  return Status::OK();
}

template <class Tdata, typename FuncT>
Status ScatterData(
    const FuncT& func,
    const Tensor* data_input, const std::vector<int64_t>& indices_data, const Tensor* updates_input, int64_t axis,
    concurrency::ThreadPool* tp,
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

  if (num_indices == 0) {
    return Status::OK();
  }

  const auto* update_data = static_cast<const Tdata*>(updates_input->DataRaw());

  // Compute outer_size (product of dims before axis) and inner_size (product of dims after axis).
  // For ScatterElements with axis=a:
  //   output[i0]...[indices[i0..iN]][...][iN] = updates[i0][...][iN]
  // Work units identified by (outer_idx, inner_idx) are completely independent:
  // they never write to the same output element, even with reductions.
  // This allows safe parallelization over outer_size * inner_size work units.
  int64_t outer_size = 1;
  for (int64_t i = 0; i < axis; ++i) {
    outer_size *= upd_shape[narrow<size_t>(i)];
  }
  const int64_t axis_size = upd_shape[narrow<size_t>(axis)];
  int64_t inner_size = 1;
  for (size_t i = narrow<size_t>(axis) + 1; i < num_dims; ++i) {
    inner_size *= upd_shape[i];
  }

  // Compute strides for the input/output tensor
  std::vector<int64_t> input_strides(num_dims);
  input_strides.back() = 1;
  if (num_dims > 1) {
    for (auto i = int64_t(num_dims - 2); i >= 0; --i) {
      input_strides[narrow<size_t>(i)] = input_data_shape[SafeInt<size_t>(i) + 1] * input_strides[SafeInt<size_t>(i) + 1];
    }
  }

  // Compute strides for the updates/indices tensor
  std::vector<int64_t> upd_strides(num_dims);
  upd_strides.back() = 1;
  if (num_dims > 1) {
    for (auto i = int64_t(num_dims - 2); i >= 0; --i) {
      upd_strides[narrow<size_t>(i)] = upd_shape[SafeInt<size_t>(i) + 1] * upd_strides[SafeInt<size_t>(i) + 1];
    }
  }

  const int64_t total_work_units = outer_size * inner_size;
  const int64_t input_axis_stride = input_strides[narrow<size_t>(axis)];
  const int64_t upd_axis_stride = upd_strides[narrow<size_t>(axis)];

  // Parallelize over independent work units.
  // Each work unit processes axis_size elements along the scatter axis.
  // Cost per unit is proportional to axis_size (number of scatter ops per work unit).
  concurrency::ThreadPool::TryParallelFor(
      tp, narrow<std::ptrdiff_t>(total_work_units), static_cast<double>(axis_size),
      [&](std::ptrdiff_t first, std::ptrdiff_t last) {
        for (std::ptrdiff_t work_idx = first; work_idx < last; ++work_idx) {
          // Decompose work_idx into outer_idx and inner_idx
          const int64_t outer_idx = static_cast<int64_t>(work_idx) / inner_size;
          const int64_t inner_idx = static_cast<int64_t>(work_idx) % inner_size;

          // Compute the base offset in the output for dimensions outside the axis.
          // For dims before axis: determined by outer_idx
          // For dims after axis: determined by inner_idx
          int64_t dst_base_offset = 0;
          int64_t outer_remain = outer_idx;
          for (int64_t d = axis - 1; d >= 0; --d) {
            const auto dim_size = upd_shape[narrow<size_t>(d)];
            const auto coord = outer_remain % dim_size;
            outer_remain /= dim_size;
            dst_base_offset += coord * input_strides[narrow<size_t>(d)];
          }
          int64_t inner_remain = inner_idx;
          for (int64_t d = int64_t(num_dims) - 1; d > axis; --d) {
            const auto dim_size = upd_shape[narrow<size_t>(d)];
            const auto coord = inner_remain % dim_size;
            inner_remain /= dim_size;
            dst_base_offset += coord * input_strides[narrow<size_t>(d)];
          }

          // Compute the base index into the updates/indices flat array
          int64_t upd_base_offset = 0;
          outer_remain = outer_idx;
          for (int64_t d = axis - 1; d >= 0; --d) {
            const auto dim_size = upd_shape[narrow<size_t>(d)];
            const auto coord = outer_remain % dim_size;
            outer_remain /= dim_size;
            upd_base_offset += coord * upd_strides[narrow<size_t>(d)];
          }
          inner_remain = inner_idx;
          for (int64_t d = int64_t(num_dims) - 1; d > axis; --d) {
            const auto dim_size = upd_shape[narrow<size_t>(d)];
            const auto coord = inner_remain % dim_size;
            inner_remain /= dim_size;
            upd_base_offset += coord * upd_strides[narrow<size_t>(d)];
          }

          // Process axis_size elements along the axis
          for (int64_t a = 0; a < axis_size; ++a) {
            const int64_t upd_flat_idx = upd_base_offset + a * upd_axis_stride;
            const int64_t axis_idx = indices_data[narrow<size_t>(upd_flat_idx)];
            const int64_t dst_offset = dst_base_offset + axis_idx * input_axis_stride;
            func(dst_base + dst_offset, update_data + upd_flat_idx);
          }
        }
      });

  return Status::OK();
}

template <typename TData>
struct ScatterDataDispatchTarget {
  Status operator()(const Tensor* data_input, const std::vector<int64_t>& indices_data, const Tensor* updates_input, int64_t axis,
                    const std::string& reduction, concurrency::ThreadPool* tp, Tensor* data_output) const {
    if (reduction == "add")
      return ScatterData<TData>(
          Func_Add<TData>(), data_input, indices_data, updates_input, axis, tp, data_output);
    else if (reduction == "mul")
      return ScatterData<TData>(
          Func_Mul<TData>(), data_input, indices_data, updates_input, axis, tp, data_output);
    else if (reduction == "min")
      return ScatterData<TData>(
          Func_Min<TData>(), data_input, indices_data, updates_input, axis, tp, data_output);
    else if (reduction == "max")
      return ScatterData<TData>(
          Func_Max<TData>(), data_input, indices_data, updates_input, axis, tp, data_output);
    else  // if (reduction == "none")
      return ScatterData<TData>(
          Func_Assignment<TData>(), data_input, indices_data, updates_input, axis, tp, data_output);
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
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();

  if (index_type == utils::ToTensorProtoElementType<int32_t>()) {
    status = GetIndices<int32_t>(*data_input, *indices_input, axis, tp, indices_data);
  } else if (index_type == utils::ToTensorProtoElementType<int64_t>()) {
    status = GetIndices<int64_t>(*data_input, *indices_input, axis, tp, indices_data);
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
      data_input, indices_data, updates_input, axis, this->reduction_, tp, data_output);

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
  ORT_RETURN_IF_ERROR(GetIndices<Tin>(*data_output, *indices_input, axis, nullptr, indices_data));
  return ScatterData<Tdata>(Func_Add<Tdata>(), data_output, indices_data, updates_input, axis, nullptr, data_output);
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

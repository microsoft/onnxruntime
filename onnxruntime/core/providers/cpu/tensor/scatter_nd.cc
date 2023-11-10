// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/scatter_nd.h"

#include "core/framework/element_type_lists.h"
#include "core/framework/op_kernel_type_control_utils.h"
#include "core/platform/threadpool.h"
#include "core/providers/op_kernel_type_control.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {

namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_DEFAULT_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterND, Input, 0,
    element_type_lists::All);
}

using EnabledScatterNDDataTypes = ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST_ALL_OPSETS(
    kCpuExecutionProvider, kOnnxDomain, ScatterND, Input, 0);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterND,
    11,
    12,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterNDDataTypes>()),
    ScatterND);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterND,
    13,
    15,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterNDDataTypes>()),
    ScatterND);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ScatterND,
    16,
    17,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterNDDataTypes>()),
    ScatterND);

ONNX_CPU_OPERATOR_KERNEL(
    ScatterND,
    18,
    KernelDefBuilder()
        .TypeConstraint("T",
                        BuildKernelDefConstraintsFromTypeList<EnabledScatterNDDataTypes>()),
    ScatterND);

Status ScatterND::ValidateShapes(
    const TensorShape& input_shape,
    const TensorShape& indice_shape,
    const TensorShape& update_shape) {
  auto input_rank = input_shape.NumDimensions();
  auto indice_rank = indice_shape.NumDimensions();
  auto update_rank = update_shape.NumDimensions();

  if (input_rank == 0 || indice_rank == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input tensor and indices tensor must has rank larger than 0. ",
                           "input shape: ", input_shape, ", indices shape: ", indice_shape);
  }

  auto last_indice_dimension = indice_shape[indice_rank - 1];
  if (last_indice_dimension > static_cast<int64_t>(input_rank)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "last dimension of indices must not be larger than rank of input tensor");
  }

  bool is_update_shape_invalid = [&]() {
    // Validate rank of update tensor
    // Per spec, the rank of the update tensor should be:
    // (Rank of input tensor) + (Rank of indices tensor) -1 - last_indice_dimension
    if (update_rank != (input_rank + indice_rank - 1 - static_cast<ptrdiff_t>(last_indice_dimension))) {
      return true;
    }

    // Validate shape of the update tensor
    // Part 1: The shape of the update tensor upto the indices rank - 1 (exclusive)
    // should match the shape of the indices tensor upto indices rank - 1 (exclusive)
    if (indice_shape.Slice(0, indice_rank - 1) != update_shape.Slice(0, indice_rank - 1)) {
      return true;
    }

    // Part 2: The shape of the update tensor after indices rank - 1 (inclusive)
    // should match the shape of the input tensor after `last_indice_dimension`
    if (input_shape.Slice(onnxruntime::narrow<size_t>(last_indice_dimension)) != update_shape.Slice(indice_rank - 1)) {
      return true;
    }

    return false;
  }();

  if (is_update_shape_invalid) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "updates tensor should have shape equal to indices.shape[:-1] + data.shape[indices.shape[-1]:]. ",
                           "updates shape: ", update_shape, ", indices shape: ", indice_shape, ", data shape: ", input_shape);
  }

  return Status::OK();
}

namespace scatternd_internal {
template <typename TData>
struct Prepare {
  const TData* input_base;
  TData* output_base;
  uint64_t element_to_copy;
  std::vector<uint64_t> element_offsets;

  Prepare() : input_base(nullptr),
              output_base(nullptr),
              element_to_copy(0),
              element_offsets(0) {}
};  // struct Prepare

template <typename TData>
Status PrepareForCompute(OpKernelContext* context, Prepare<TData>& p) {
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto* indice_tensor = context->Input<Tensor>(1);
  const auto* update_tensor = context->Input<Tensor>(2);

  const auto& input_shape = input_tensor->Shape();
  const auto& indice_shape = indice_tensor->Shape();
  const auto& update_shape = update_tensor->Shape();

  ORT_RETURN_IF_ERROR(ScatterND::ValidateShapes(input_shape, indice_shape, update_shape));

  auto output_tensor = context->Output(0, input_shape);

  const auto* src_base = input_tensor->Data<TData>();
  auto* dst_base = output_tensor->MutableData<TData>();
  const bool is_string_type = input_tensor->IsDataTypeString();

  auto last_indice_dimension = indice_shape[indice_shape.NumDimensions() - 1];

  // Re-use input for output. If input/output Tensor* are the same, do not copy.
  if (src_base != dst_base) {
    if (is_string_type) {
      const auto* str_begin = input_tensor->Data<std::string>();
      const std::string* str_end = str_begin + input_shape.Size();
      auto* dst = output_tensor->MutableData<std::string>();
      std::copy(str_begin, str_end, dst);
    } else {
      memcpy((void*)dst_base, (const void*)src_base, input_tensor->SizeInBytes());
    }
  }

  std::vector<int64_t> element_counts(onnxruntime::narrow<size_t>(last_indice_dimension), 0LL);  // Number of elements for each input dimension

  TensorPitches input_strides(input_shape);
  for (int64_t i = 0; i < last_indice_dimension; ++i) {
    element_counts[onnxruntime::narrow<size_t>(i)] = input_strides[onnxruntime::narrow<size_t>(i)];
  }

  p.element_to_copy = input_shape.SizeFromDimension(onnxruntime::narrow<size_t>(last_indice_dimension));
  const int64_t* indice_offset = indice_tensor->Data<int64_t>();
  auto offset_count = indice_shape.Size() / last_indice_dimension;  // Times to copy
  p.element_offsets.assign(onnxruntime::narrow<size_t>(offset_count), 0LL);

  p.input_base = update_tensor->Data<TData>();
  p.output_base = output_tensor->MutableData<TData>();

  for (int64_t i = 0; i < offset_count; ++i) {
    for (int64_t j = 0; j < last_indice_dimension; ++j) {
      auto indice = *(indice_offset + i * last_indice_dimension + j);

      if (indice >= 0) {
        if (indice >= input_shape[onnxruntime::narrow<size_t>(j)]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid indice found, indice = ", indice);
        }
      } else {
        if (indice < -input_shape[onnxruntime::narrow<size_t>(j)]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "invalid indice found, indice = ", indice);
        } else {
          indice += input_shape[onnxruntime::narrow<size_t>(j)];
        }
      }

      p.element_offsets[onnxruntime::narrow<size_t>(i)] += indice * element_counts[onnxruntime::narrow<size_t>(j)];
    }
  }
  return Status::OK();
}

template <class T>
struct Func_Copy_ND {
  void operator()(T* a, const T* b, uint64_t element_to_copy) const {
    memcpy(a, b, SafeInt<size_t>(element_to_copy) * sizeof(T));
  }
};

template <>
struct Func_Copy_ND<std::string> {
  void operator()(std::string* a, const std::string* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0)
      (*a++) = (*b++);
  }
};

template <class T>
struct Func_Add_ND {
  void operator()(T* a, const T* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0)
      (*a++) += (*b++);
  }
};

template <>
struct Func_Add_ND<bool> {
  void operator()(bool* a, const bool* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0)
      (*a++) |= (*b++);
  }
};

template <>
struct Func_Add_ND<MLFloat16> {
  void operator()(MLFloat16*, const MLFloat16*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: MLFloat16 data type is not supported with ScatterND opset 16 when reduction is 'add'.");
  }
};

template <>
struct Func_Add_ND<BFloat16> {
  void operator()(BFloat16*, const BFloat16*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterND opset 16 when reduction is 'add'.");
  }
};

template <class T>
struct Func_Mul_ND {
  void operator()(T* a, const T* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0)
      (*a++) *= (*b++);
  }
};

template <>
struct Func_Mul_ND<bool> {
  void operator()(bool* a, const bool* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0)
      (*a++) &= (*b++);
  }
};

template <>
struct Func_Mul_ND<std::string> {
  void operator()(std::string*, const std::string*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterND opset 16 when reduction is 'mul'.");
  }
};

template <>
struct Func_Mul_ND<MLFloat16> {
  void operator()(MLFloat16*, const MLFloat16*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: MLFloat16 data type is not supported with ScatterND opset 16 when reduction is 'mul'.");
  }
};

template <>
struct Func_Mul_ND<BFloat16> {
  void operator()(BFloat16*, const BFloat16*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterND opset 16 when reduction is 'mul'.");
  }
};

template <class T>
struct Func_Min_ND {
  void operator()(T* a, const T* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0) {
      (*a) = (*a) < (*b) ? (*a) : (*b);
      a++;
      b++;
    }
  }
};

template <>
struct Func_Min_ND<bool> {
  void operator()(bool*, const bool*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: bool data type is not supported with ScatterND opset 18 when reduction is 'min'.");
  }
};

template <>
struct Func_Min_ND<std::string> {
  void operator()(std::string*, const std::string*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterND opset 18 when reduction is 'min'.");
  }
};

template <>
struct Func_Min_ND<MLFloat16> {
  void operator()(MLFloat16*, const MLFloat16*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: MLFloat16 data type is not supported with ScatterND opset 18 when reduction is 'min'.");
  }
};

template <>
struct Func_Min_ND<BFloat16> {
  void operator()(BFloat16*, const BFloat16*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterND opset 18 when reduction is 'min'.");
  }
};

template <class T>
struct Func_Max_ND {
  void operator()(T* a, const T* b, uint64_t element_to_copy) const {
    while (element_to_copy-- > 0) {
      (*a) = (*a) > (*b) ? (*a) : (*b);
      a++;
      b++;
    }
  }
};

template <>
struct Func_Max_ND<bool> {
  void operator()(bool*, const bool*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: bool data type is not supported with ScatterND opset 18 when reduction is 'max'.");
  }
};

template <>
struct Func_Max_ND<std::string> {
  void operator()(std::string*, const std::string*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: string data type is not supported with ScatterND opset 18 when reduction is 'max'.");
  }
};

template <>
struct Func_Max_ND<MLFloat16> {
  void operator()(MLFloat16*, const MLFloat16*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: MLFloat16 data type is not supported with ScatterND opset 18 when reduction is 'max'.");
  }
};

template <>
struct Func_Max_ND<BFloat16> {
  void operator()(BFloat16*, const BFloat16*, uint64_t) const {
    ORT_NOT_IMPLEMENTED("CPU execution provider: BFloat16 data type is not supported with ScatterND opset 18 when reduction is 'max'.");
  }
};

template <typename TData>
struct ScatterNDDispatchTarget {
  Status operator()(OpKernelContext* context, concurrency::ThreadPool* tp, ScatterND::Reduction reduction) const {
    Prepare<TData> prepare;
    ORT_RETURN_IF_ERROR(PrepareForCompute(context, prepare));

    auto lambda = [&](int64_t i) {
      switch (reduction) {
        case ScatterND::Reduction::Add: {
          auto func = Func_Add_ND<TData>();
          func(
              prepare.output_base + prepare.element_offsets[onnxruntime::narrow<size_t>(i)],
              prepare.input_base + i * prepare.element_to_copy,
              prepare.element_to_copy);
        } break;
        case ScatterND::Reduction::Mul: {
          auto func = Func_Mul_ND<TData>();
          func(
              prepare.output_base + prepare.element_offsets[onnxruntime::narrow<size_t>(i)],
              prepare.input_base + i * prepare.element_to_copy,
              prepare.element_to_copy);
        } break;
        case ScatterND::Reduction::Min: {
          auto func = Func_Min_ND<TData>();
          func(
              prepare.output_base + prepare.element_offsets[onnxruntime::narrow<size_t>(i)],
              prepare.input_base + i * prepare.element_to_copy,
              prepare.element_to_copy);
        } break;
        case ScatterND::Reduction::Max: {
          auto func = Func_Max_ND<TData>();
          func(
              prepare.output_base + prepare.element_offsets[onnxruntime::narrow<size_t>(i)],
              prepare.input_base + i * prepare.element_to_copy,
              prepare.element_to_copy);
        } break;
        default:
        case ScatterND::Reduction::None: {
          auto func = Func_Copy_ND<TData>();
          func(
              prepare.output_base + prepare.element_offsets[onnxruntime::narrow<size_t>(i)],
              prepare.input_base + i * prepare.element_to_copy,
              prepare.element_to_copy);
        } break;
      }
    };
    concurrency::ThreadPool::TryParallelFor(
        tp, prepare.element_offsets.size(), static_cast<double>(prepare.element_to_copy),
        [&lambda](ptrdiff_t first, ptrdiff_t last) {
          for (int i = static_cast<int>(first), end = static_cast<int>(last); i < end; ++i) {
            lambda(i);
          }
        });
    return Status::OK();
  }
};
}  // namespace scatternd_internal

Status ScatterND::Compute(OpKernelContext* context) const {
  using namespace scatternd_internal;
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  const auto data_type = context->Input<Tensor>(0)->GetElementType();
  utils::MLTypeCallDispatcherFromTypeList<EnabledScatterNDDataTypes> dispatcher{data_type};
  Status status = dispatcher.template InvokeRet<Status, ScatterNDDispatchTarget>(context, tp, this->reduction_);
  return status;
}
}  // namespace onnxruntime

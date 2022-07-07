// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/gather_elements.h"

#include "core/providers/cuda/tensor/gather_elements_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(GatherElements, kOnnxDomain, 13, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create())
                            .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                                            DataTypeImpl::GetTensorType<int64_t>()}),
                        GatherElements);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(GatherElements, kOnnxDomain, 11, 12, kCudaExecutionProvider,
                                  (*KernelDefBuilder::Create())
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .TypeConstraint("Tind",
                                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                                              DataTypeImpl::GetTensorType<int64_t>()}),
                                  GatherElements);

namespace {
bool CanSkip(const TensorShapeVector& input_shape, const TensorShapeVector& indices_shape, size_t dim) {
  return input_shape[dim] == 1 && indices_shape[dim] == 1;
}

bool CanMerge(const TensorShapeVector& input_shape, const TensorShapeVector& indices_shape, size_t src, size_t dst) {
  return input_shape[src] == indices_shape[src] && input_shape[dst] == indices_shape[dst];
}

void Move(TensorShapeVector& input_shape, TensorShapeVector& indices_shape, size_t src, size_t dst) {
  input_shape[dst] = input_shape[src];
  indices_shape[dst] = indices_shape[src];
}

void Merge(TensorShapeVector& input_shape, TensorShapeVector& indices_shape, size_t src, size_t dst) {
  input_shape[dst] *= input_shape[src];
  indices_shape[dst] *= indices_shape[src];
}
}  // namespace

void CoalesceDimensions(TensorShapeVector& input_shape, TensorShapeVector& indices_shape, int64_t axis,
                        int64_t& new_axis, int64_t& new_rank, int64_t& input_stride_along_axis,
                        TArray<int64_t>& masked_input_strides, TArray<fast_divmod>& indices_fdms) {
  size_t rank = input_shape.size();
  // Reverse for better calculation.
  std::reverse(input_shape.begin(), input_shape.end());
  std::reverse(indices_shape.begin(), indices_shape.end());
  if (axis < 0 || axis >= static_cast<int64_t>(rank)) ORT_THROW("Invalid axis in CoalesceDimensions: ", axis);
  size_t reverse_axis = rank - 1 - static_cast<size_t>(axis);
  size_t curr = 0, next = 0;
  while (curr < reverse_axis && CanSkip(input_shape, indices_shape, curr)) ++curr;
  if (curr < reverse_axis) {
    if (curr > 0) Move(input_shape, indices_shape, curr, 0);
    next = curr + 1;
    curr = 0;
    while (next < reverse_axis) {
      if (!CanSkip(input_shape, indices_shape, next)) {
        if (CanMerge(input_shape, indices_shape, next, curr)) {
          Merge(input_shape, indices_shape, next, curr);
        } else {
          ++curr;
          if (curr != next) Move(input_shape, indices_shape, next, curr);
        }
      }
      ++next;
    }
  }

  if (curr == reverse_axis) {
    if (curr > 0) Move(input_shape, indices_shape, curr, 0);
    curr = 0;
  } else {
    ++curr;
    // next is now the reverse_axis
    if (curr != next) Move(input_shape, indices_shape, next, curr);
  }
  size_t new_reverse_axis = curr;
  next = reverse_axis + 1;
  while (next < rank && CanSkip(input_shape, indices_shape, next)) ++next;
  if (next < rank) {
    ++curr;
    if (curr != next) Move(input_shape, indices_shape, next, curr);
    ++next;
    while (next < rank) {
      if (!CanSkip(input_shape, indices_shape, next)) {
        if (CanMerge(input_shape, indices_shape, next, curr)) {
          Merge(input_shape, indices_shape, next, curr);
        } else {
          ++curr;
          if (curr != next) Move(input_shape, indices_shape, next, curr);
        }
      }
      ++next;
    }
  }

  new_rank = curr + 1;
  new_axis = static_cast<int64_t>(new_rank - 1 - new_reverse_axis);
  input_shape.resize(new_rank);
  std::reverse(input_shape.begin(), input_shape.end());
  indices_shape.resize(new_rank);
  std::reverse(indices_shape.begin(), indices_shape.end());

  // Set stride along axis to 0 so we don't need IF statement to check in kernel.
  TensorPitches masked_input_strides_vec(input_shape);
  input_stride_along_axis = masked_input_strides_vec[new_axis];
  masked_input_strides_vec[new_axis] = 0;
  int32_t new_rank_32bit = static_cast<int32_t>(new_rank);
  masked_input_strides.SetSize(new_rank_32bit);
  indices_fdms.SetSize(new_rank_32bit);
  TensorPitches indices_strides(indices_shape);
  for (auto i = 0; i < new_rank_32bit; ++i) {
    masked_input_strides[i] = masked_input_strides_vec[i];
    indices_fdms[i] = fast_divmod(gsl::narrow_cast<int>(indices_strides[i]));
  }
}

// GatherElementsGrad needs atomic_add which supports float types only, so use half, float and double for 16, 32, and 64
// bits data respectively.
ONNX_NAMESPACE::TensorProto_DataType GetElementType(size_t element_size) {
  switch (element_size) {
    case sizeof(int8_t):
      return ONNX_NAMESPACE::TensorProto_DataType_INT8;
    case sizeof(MLFloat16):
      return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
    case sizeof(float):
      return ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
    case sizeof(double):
      return ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
    // should not reach here as we validate if the all relevant types are supported in the Compute method
    default:
      return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  }
}

#define CASE_GATHER_ELEMENTS_IMPL(type)                                                               \
  case sizeof(type): {                                                                                \
    const type* indices_data = reinterpret_cast<const type*>(indices_data_raw);                       \
    GatherElementsImpl(stream, rank, axis, input_data, input_dim_along_axis, input_stride_along_axis, \
                       masked_input_strides, indices_data, indices_size, indices_fdms, output_data);  \
  } break

template <typename T>
struct GatherElements::ComputeImpl {
  Status operator()(cudaStream_t stream, const void* input_data_raw, const void* indices_data_raw,
                    void* output_data_raw, const int64_t rank, const int64_t axis, const int64_t input_dim_along_axis,
                    const int64_t input_stride_along_axis, TArray<int64_t>& masked_input_strides,
                    const int64_t indices_size, TArray<fast_divmod>& indices_fdms,
                    const size_t index_element_size) const {
    typedef typename ToCudaType<T>::MappedType CudaT;
    const CudaT* input_data = reinterpret_cast<const CudaT*>(input_data_raw);
    CudaT* output_data = reinterpret_cast<CudaT*>(output_data_raw);
    switch (index_element_size) {
      CASE_GATHER_ELEMENTS_IMPL(int32_t);
      CASE_GATHER_ELEMENTS_IMPL(int64_t);
      // should not reach here as we validate if the all relevant types are supported in the Compute method
      default:
        ORT_THROW("Unsupported indices element size by the GatherElements CUDA kernel");
    }
    return Status::OK();
  }
};

#undef CASE_GATHER_ELEMENTS_IMPL

Status GatherElements::ComputeInternal(OpKernelContext* context) const {
  // Process input data tensor
  const auto* input_tensor = context->Input<Tensor>(0);
  const auto& input_shape = input_tensor->Shape();
  const int64_t input_rank = static_cast<int64_t>(input_shape.NumDimensions());

  // Process indices tensor
  const auto* indices_tensor = context->Input<Tensor>(1);
  const auto& indices_shape = indices_tensor->Shape();
  const int64_t indices_size = indices_shape.Size();

  // Handle negative axis if any
  const int64_t axis = HandleNegativeAxis(axis_, input_rank);

  // Validate input shapes and ranks (invoke the static method in the CPU GatherElements kernel that hosts the shared
  // checks)
  ORT_RETURN_IF_ERROR(onnxruntime::GatherElements::ValidateInputShapes(input_shape, indices_shape, axis));

  // create output tensor
  auto* output_tensor = context->Output(0, indices_shape);

  // if there are no elements in 'indices' - nothing to process
  if (indices_size == 0) return Status::OK();

  TensorShapeVector input_shape_vec = input_shape.AsShapeVector();
  TensorShapeVector indices_shape_vec = indices_shape.AsShapeVector();
  int64_t new_axis, new_rank, input_stride_along_axis;
  TArray<int64_t> masked_input_strides;
  TArray<fast_divmod> indices_fdms;
  CoalesceDimensions(input_shape_vec, indices_shape_vec, axis, new_axis, new_rank, input_stride_along_axis,
                     masked_input_strides, indices_fdms);

  // Use element size instead of concrete types so we can specialize less template functions to reduce binary size.
  int dtype = GetElementType(input_tensor->DataType()->Size());
  if (dtype == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
    ORT_THROW("Unsupported element size by the GatherElements CUDA kernel");
  }

  utils::MLTypeCallDispatcher<int8_t, MLFloat16, float, double> t_disp(dtype);
  return t_disp.InvokeRet<Status, ComputeImpl>(Stream(), input_tensor->DataRaw(), indices_tensor->DataRaw(),
                                               output_tensor->MutableDataRaw(), new_rank, new_axis,
                                               input_shape_vec[new_axis], input_stride_along_axis, masked_input_strides,
                                               indices_size, indices_fdms, indices_tensor->DataType()->Size());
}

}  // namespace cuda
}  // namespace onnxruntime

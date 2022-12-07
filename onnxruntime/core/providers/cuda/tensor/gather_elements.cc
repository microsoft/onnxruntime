// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/gather_elements.h"

#include "core/providers/cuda/tensor/gather_elements_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

// Ideally both input and indices can support strided tensor, for training case, the indices is the input for both
// GatherElements and GatherElementsGrad, indices supporting strided tensor is more useful for saving memory.
// So we only mark indices as MayStridedInput for now. Will do this for input once needed.
#ifdef ENABLE_STRIDED_TENSORS
#define CREATE_GATHER_ELEMENTS_KERNEL_DEF (*KernelDefBuilder::Create()).MayStridedInput(1)
#else
#define CREATE_GATHER_ELEMENTS_KERNEL_DEF (*KernelDefBuilder::Create())
#endif

ONNX_OPERATOR_KERNEL_EX(GatherElements, kOnnxDomain, 13, kCudaExecutionProvider,
                        CREATE_GATHER_ELEMENTS_KERNEL_DEF.TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                            .TypeConstraint("Tind", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                                            DataTypeImpl::GetTensorType<int64_t>()}),
                        GatherElements);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(GatherElements, kOnnxDomain, 11, 12, kCudaExecutionProvider,
                                  CREATE_GATHER_ELEMENTS_KERNEL_DEF
                                      .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
                                      .TypeConstraint("Tind",
                                                      std::vector<MLDataType>{DataTypeImpl::GetTensorType<int32_t>(),
                                                                              DataTypeImpl::GetTensorType<int64_t>()}),
                                  GatherElements);

#undef CREATE_GATHER_ELEMENTS_KERNEL_DEF

void CoalesceDimensions(TensorShapeVector& input_shape, TensorShapeVector& indices_shape,
                        TensorShapeVector* p_indices_strides, int64_t axis, GatherScatterElementsArgs& args) {
  size_t rank = input_shape.size();
  if (axis < 0 || axis >= static_cast<int64_t>(rank)) ORT_THROW("Invalid axis in CoalesceDimensions: ", axis);
  size_t new_axis = static_cast<size_t>(axis);
  auto CanCoalesce = [&](size_t dst, size_t src) {
    if (dst == static_cast<size_t>(new_axis) || src == static_cast<size_t>(new_axis)) return false;
    if (input_shape[dst] == 1 && indices_shape[dst] == 1) return true;
    if (input_shape[src] == 1 && indices_shape[src] == 1) return true;
    return input_shape[dst] == indices_shape[dst] && input_shape[src] == indices_shape[src] &&
           (!p_indices_strides || (*p_indices_strides)[dst] == indices_shape[src] * (*p_indices_strides)[src]);
  };

  size_t curr = 0;
  for (size_t next = 1; next < rank; ++next) {
    if (CanCoalesce(curr, next)) {
      if (indices_shape[next] != 1 && p_indices_strides) {
        (*p_indices_strides)[curr] = (*p_indices_strides)[next];
      }
      input_shape[curr] *= input_shape[next];
      indices_shape[curr] *= indices_shape[next];
    } else {
      if (next == static_cast<size_t>(new_axis)) {
        // Handle all dims outside of axis are 1-dim.
        if (input_shape[curr] != 1 || indices_shape[curr] != 1) ++curr;
        new_axis = static_cast<int64_t>(curr);
      } else {
        ++curr;
      }
      if (curr != next) {
        input_shape[curr] = input_shape[next];
        indices_shape[curr] = indices_shape[next];
        if (p_indices_strides) (*p_indices_strides)[curr] = (*p_indices_strides)[next];
      }
    }
  }
  // Handle all dims inside of axis are 1-dim.
  if (curr > static_cast<size_t>(new_axis) && input_shape[curr] == 1 && indices_shape[curr] == 1) {
    --curr;
  }

  size_t new_rank = curr + 1;
  args.rank = static_cast<int64_t>(new_rank);
  args.axis = static_cast<int64_t>(new_axis);
  input_shape.resize(new_rank);
  indices_shape.resize(new_rank);
  if (p_indices_strides) {
    p_indices_strides->resize(new_rank);
  }

  // Set stride along axis to 0 so we don't need IF statement to check in kernel.
  TensorPitches masked_input_strides_vec(input_shape);
  args.input_stride_along_axis = masked_input_strides_vec[args.axis];
  args.input_dim_along_axis = input_shape[args.axis];
  masked_input_strides_vec[args.axis] = 0;
  args.masked_input_strides = TArray<int64_t>(ToConstSpan(masked_input_strides_vec));
  args.indices_fdms.SetSize(static_cast<int32_t>(new_rank));
  TensorPitches indices_shape_strides(indices_shape);
  for (int32_t i = 0; i < static_cast<int32_t>(new_rank); ++i) {
    args.indices_fdms[i] = fast_divmod(gsl::narrow_cast<int>(indices_shape_strides[i]));
  }
  if (p_indices_strides) {
    args.indices_strides = TArray<int64_t>(ToConstSpan(*p_indices_strides));
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

#define CASE_GATHER_ELEMENTS_IMPL(type)                                         \
  case sizeof(type): {                                                          \
    const type* indices_data = reinterpret_cast<const type*>(indices_data_raw); \
    GatherElementsImpl(stream, input_data, indices_data, output_data, args);    \
  } break

template <typename T>
struct GatherElements::ComputeImpl {
  Status operator()(cudaStream_t stream, const void* input_data_raw, const void* indices_data_raw,
                    void* output_data_raw, const size_t index_element_size,
                    const GatherScatterElementsArgs& args) const {
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

  GatherScatterElementsArgs args;
  args.indices_size = indices_size;
  TensorShapeVector input_shape_vec = input_shape.AsShapeVector();
  TensorShapeVector indices_shape_vec = indices_shape.AsShapeVector();
  TensorShapeVector* p_indices_strides_vec = nullptr;
  TensorShapeVector indices_strides_vec;
#ifdef ENABLE_STRIDED_TENSORS
  if (!indices_tensor->IsContiguous()) {
    indices_strides_vec = ToShapeVector(indices_tensor->Strides());
    p_indices_strides_vec = &indices_strides_vec;
  }
#endif
  CoalesceDimensions(input_shape_vec, indices_shape_vec, p_indices_strides_vec, axis, args);

  // Use element size instead of concrete types so we can specialize less template functions to reduce binary size.
  int dtype = GetElementType(input_tensor->DataType()->Size());
  if (dtype == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
    ORT_THROW("Unsupported element size by the GatherElements CUDA kernel");
  }

  utils::MLTypeCallDispatcher<int8_t, MLFloat16, float, double> t_disp(dtype);
  return t_disp.InvokeRet<Status, ComputeImpl>(Stream(), input_tensor->DataRaw(), indices_tensor->DataRaw(),
                                               output_tensor->MutableDataRaw(), indices_tensor->DataType()->Size(),
                                               args);
}

}  // namespace cuda
}  // namespace onnxruntime

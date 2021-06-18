// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nonzero_op.h"
#include "nonzero_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

// kernel builder functions
#define NONZERO_TYPED_KERNEL_WITH_TYPE_NAME(type, type_name)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                                                    \
      NonZero,                                                                                \
      kOnnxDomain,                                                                            \
      9, 12,                                                                                  \
      type_name,                                                                              \
      kCudaExecutionProvider,                                                                 \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      NonZero<type>)                                                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                              \
      NonZero,                                                                                \
      kOnnxDomain,                                                                            \
      13,                                                                                     \
      type_name,                                                                              \
      kCudaExecutionProvider,                                                                 \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<type>()), \
      NonZero<type>)

#define NONZERO_TYPED_KERNEL(type) \
  NONZERO_TYPED_KERNEL_WITH_TYPE_NAME(type, type)

// start with a subset of types, enable more as needed...
NONZERO_TYPED_KERNEL(bool)
NONZERO_TYPED_KERNEL(uint8_t)
//NONZERO_TYPED_KERNEL(uint16_t)
//NONZERO_TYPED_KERNEL(uint32_t)
//NONZERO_TYPED_KERNEL(uint64_t)
//NONZERO_TYPED_KERNEL(int8_t)
//NONZERO_TYPED_KERNEL(int16_t)
NONZERO_TYPED_KERNEL(int32_t)
NONZERO_TYPED_KERNEL(int64_t)
NONZERO_TYPED_KERNEL(MLFloat16)
//NONZERO_TYPED_KERNEL(BFloat16)
NONZERO_TYPED_KERNEL(float)
//NONZERO_TYPED_KERNEL(double)
//NONZERO_TYPED_KERNEL_WITH_TYPE_NAME(std::string, string)

#undef NONZERO_TYPED_KERNEL
#undef NONZERO_TYPED_KERNEL_WITH_TYPE_NAME

template <typename T>
Status NonZero<T>::ComputeInternal(OpKernelContext* context) const {
  static const std::vector<int64_t> kScalarDims = {1};
  const auto x = context->Input<Tensor>(0);

  int nonzero_elements = 0;
  const auto& x_shape = x->Shape();
  const int x_rank = x_shape.IsScalar() ? 1 : static_cast<int>(x_shape.NumDimensions());
  const std::vector<int64_t>& x_dims = (x_shape.IsScalar()) ? kScalarDims : x_shape.GetDims();
  const int64_t x_size = x_shape.Size();
  if (x_size > 0) {
    auto x_data = reinterpret_cast<const typename ToCudaType<T>::MappedType*>(x->template Data<T>());

    const int number_of_blocks = NonZeroCalcBlockCount(x_size);
    auto prefix_buffer = GetScratchBuffer<int>(number_of_blocks);
    int* prefix_counts = prefix_buffer.get();
    CUDA_RETURN_IF_ERROR(NonZeroCountEachBlock(Stream(), x_data, x_size, prefix_counts));

    size_t temp_storage_bytes = 0;
    CUDA_RETURN_IF_ERROR(NonZeroCalcPrefixSumTempStorageBytes(Stream(), prefix_counts, number_of_blocks, temp_storage_bytes));
    auto temp_buffer = GetScratchBuffer<uint8_t>(temp_storage_bytes);
    auto d_temp_storage = temp_buffer.get();
    CUDA_RETURN_IF_ERROR(NonZeroInclusivePrefixSum(Stream(), d_temp_storage, temp_storage_bytes, prefix_counts, number_of_blocks));

    // cudaMemcpyAsync from device memory to pageable host memory will return only once the copy has completed.
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        &nonzero_elements, prefix_counts + number_of_blocks - 1,
        sizeof(int), cudaMemcpyDeviceToHost, Stream()));

    TArray<fast_divmod> fdm_x_strides(x_rank);
    TensorPitches x_strides(x_dims);
    for (auto i = 0; i < x_rank; i++) {
      fdm_x_strides[i] = fast_divmod(static_cast<int>(x_strides[i]));
    }

    auto* output_tensor = context->Output(0, {x_rank, nonzero_elements});
    ORT_ENFORCE(output_tensor, "failed to get first output!");
    CUDA_RETURN_IF_ERROR(NonZeroOutputPositions(
        Stream(), x_data, x_size, x_rank, fdm_x_strides,
        prefix_counts, nonzero_elements, output_tensor->template MutableData<int64_t>()));
  } else {
    context->Output(0, {x_rank, nonzero_elements});
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

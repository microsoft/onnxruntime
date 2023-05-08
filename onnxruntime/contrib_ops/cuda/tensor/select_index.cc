// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/tensor/select_index.h"
#include "contrib_ops/cuda/tensor/select_index_impl.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_SELECT_INDEX_KERNEL_TYPED(op_name, T)             \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                   \
      op_name,                                                     \
      kMSDomain,                                                   \
      1,                                                           \
      T,                                                           \
      kCudaExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                                \
          .TypeConstraint("TI", DataTypeImpl::GetTensorType<T>()), \
      SelectIndex<T>);

REGISTER_SELECT_INDEX_KERNEL_TYPED(SelectIndex, int64_t)

template <typename T>
Status SelectIndex<T>::ComputeInternal(OpKernelContext* context) const {
  static const TensorShape kScalarDims{1};
  const auto x = context->Input<Tensor>(0);

  int nonzero_elements = 0;
  const auto& x_shape = x->Shape();
  const int x_rank = x_shape.IsScalar() ? 1 : static_cast<int>(x_shape.NumDimensions());
  auto x_dims = (x_shape.IsScalar()) ? kScalarDims.GetDims() : x_shape.GetDims();
  const int64_t x_size = x_shape.Size();
  if (x_size > 0) {
    auto x_data = reinterpret_cast<const typename ToCudaType<T>::MappedType*>(x->Data<T>());

    const int number_of_blocks = NonZeroCalcBlockCount(x_size);
    auto prefix_buffer = GetScratchBuffer<int>(number_of_blocks, context->GetComputeStream());
    int* prefix_counts = prefix_buffer.get();
    CUDA_RETURN_IF_ERROR(NonZeroCountEachBlock(Stream(context), x_data, x_size, prefix_counts));

    size_t temp_storage_bytes = 0;
    CUDA_RETURN_IF_ERROR(NonZeroCalcPrefixSumTempStorageBytes(Stream(context), prefix_counts, number_of_blocks, temp_storage_bytes));
    auto temp_buffer = GetScratchBuffer<uint8_t>(temp_storage_bytes, context->GetComputeStream());
    auto d_temp_storage = temp_buffer.get();
    CUDA_RETURN_IF_ERROR(NonZeroInclusivePrefixSum(Stream(context), d_temp_storage, temp_storage_bytes, prefix_counts, number_of_blocks));

    // cudaMemcpyAsync from device memory to pageable host memory will return only once the copy has been completed.
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
        &nonzero_elements, prefix_counts + number_of_blocks - 1,
        sizeof(int), cudaMemcpyDeviceToHost, Stream(context)));

    TArray<fast_divmod> fdm_x_strides(x_rank);
    TensorPitches x_strides(x_dims);
    for (auto i = 0; i < x_rank; i++) {
      fdm_x_strides[i] = fast_divmod(static_cast<int>(x_strides[i]));
    }

    auto* output_tensor = context->Output(0, {x_rank, nonzero_elements});
    ORT_ENFORCE(output_tensor, "failed to get first output!");
    CUDA_RETURN_IF_ERROR(NonZeroOutputPositions(
        Stream(context), x_data, x_size, x_rank, fdm_x_strides,
        prefix_counts, nonzero_elements, output_tensor->MutableData<int64_t>()));
  } else {
    context->Output(0, {x_rank, nonzero_elements});
  }

  return Status::OK();
}

#undef REGISTER_SELECT_INDEX_KERNEL_TYPED

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

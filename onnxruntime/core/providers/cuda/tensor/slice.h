// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

namespace SliceCuda {

Status Impl(cudaStream_t stream,
            const void* input_data,
            const TensorShape& input_shape,
            void* output_data,
            SliceOp::PrepareForComputeMetadata& prepare_metadata,
            size_t element_size);

}  // namespace SliceCuda

template <bool dynamic>
class Slice : public CudaKernel, public SliceBase {
 public:
  Slice(const OpKernelInfo& info) : CudaKernel(info), SliceBase(info, dynamic) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  virtual const Tensor* GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const;
  virtual Status FillInputVectors(OpKernelContext* ctx, TensorShapeVector& input_starts,
                                  TensorShapeVector& input_ends, TensorShapeVector& input_axes,
                                  TensorShapeVector& input_steps) const;

  virtual Status CallSliceImp(size_t element_size, size_t dimension_count, const TArray<int64_t>& starts_buffer,
                              const TArray<int64_t>& steps_buffer, const TArray<int64_t>& input_strides,
                              const TArray<fast_divmod>& output_strides, OpKernelContext* ctx,
                              const TensorShape& output_shape) const;
};
}  // namespace cuda
}  // namespace onnxruntime

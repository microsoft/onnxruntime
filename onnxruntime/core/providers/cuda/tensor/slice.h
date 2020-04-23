// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

template <bool dynamic>
class Slice : public CudaKernel, public SliceBase {
public:
  Slice(const OpKernelInfo& info) : CudaKernel(info), SliceBase(info, dynamic) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  virtual const Tensor* GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const;
  virtual void FillInputVectors(OpKernelContext* ctx, std::vector<int64_t>& input_starts,
                                std::vector<int64_t>& input_ends, std::vector<int64_t>& input_axes,
                                std::vector<int64_t>& input_steps) const;

  virtual Status CallSliceImp(size_t element_size, size_t dimension_count, const TArray<int64_t>& starts_buffer,
                              const TArray<int64_t>& steps_buffer, const TArray<int64_t>& input_strides,
                              const TArray<fast_divmod>& output_strides, OpKernelContext* ctx,
                              TensorShape output_shape) const;
};
}  // namespace cuda
}  // namespace onnxruntime

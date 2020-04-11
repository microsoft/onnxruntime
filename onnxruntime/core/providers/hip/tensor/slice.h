// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/cpu/tensor/slice.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/hip/hip_utils.h"

namespace onnxruntime {
namespace hip {

template <bool dynamic>
class Slice : public HipKernel, public SliceBase {
 public:
  Slice(const OpKernelInfo& info) : HipKernel(info), SliceBase(info, dynamic) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;

 private:
  virtual const Tensor* GetSlicedOrUnslicedTensor(OpKernelContext* ctx) const;
  virtual void FillInputVectors(OpKernelContext* ctx, std::vector<int64_t>& input_starts,
                                std::vector<int64_t>& input_ends, std::vector<int64_t>& input_axes,
                                std::vector<int64_t>& input_steps) const;

  virtual Status CallSliceImp(size_t element_size, size_t dimension_count, const int64_t* starts_buffer,
                              const int64_t* steps_buffer, const int64_t* input_strides,
                              const fast_divmod* output_strides, OpKernelContext* ctx,
                              TensorShape output_shape) const;
};
}  // namespace hip
}  // namespace onnxruntime
